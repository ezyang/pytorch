# Profile-guided optimization for Dynamo

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import logging
import os
import os.path
import pickle
from typing import Dict, Optional, TYPE_CHECKING, TypeAlias
from typing_extensions import Self


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._dynamo.variables.builder import FrameStateSizeEntry


# How does in memory representation work?  Concretely, this module is
# responsible for holding GLOBAL state representing the state it holds, no
# other copies permitted.  So we retire frame_state entirely and store it
# here.  This should be reset when Dynamo is reset.  We never GC information
# (similar to how the filesystem doesn't get cleaned up except by tmp
# cleaner), so the expectation is the information is relatively cheap and we
# don't mind leaking it.


# How does the filesystem cache work?  We would like to account for situations
# where we have multiple processes attempting to write to the same filesystem
# at once.  We would also like the write algorithm to resemble our remote
# cache, so that there is for the most part only one implementation.
#
# Here is a simplest scheme that accounts for multiple process writing to the
# same place:
#
#   - Every "filename, firstlineno, name" gets a distinct cache directory,
#     containing possible multiple cache entries.  The cache entries
#     themselves are content addressed so that if you write the same cache
#     data multiple times you only end up with one file.
#
#   - There is a defined merge operation, which says what the semantics should
#     be when there are multiple cache entries.  The merge operation forms a
#     semilattice (it is associative, commutative and idempotent).
#
#   - Before compiling, we iterate through all entries in the directory
#     and merge them to form the cache lookup result, which we use for
#     compilation.
#
#   - If we collect a new cache state which is not equal to the previous entry,
#     we atomically write a new entry to the cache directory, so that our
#     state is incorporated into subsequent reads.
#
# As an optional optimization, cache entries can be packed into a single
# bundle via a process that reads all cache entries, atomically writes out
# their merge, and then deletes all the other cache entries they write.  But I
# did some measurements and reading out eight 1KB string pickles from a directory
# is only 120ns so as long as you make sure that the directory doesn't keep
# getting bigger and bigger (but it shouldn't, because it's content addressed),
# and it seems a bit difficult to do this update atomically, and taking out a
# full file lock is expensive 100ms with eight processes).  So we have not
# implemented this.

Hash: TypeAlias = str


def sha1_hash(data: bytes) -> str:
    # [:51] to strip off the "Q====" suffix common to every hash value.
    return base64.b32encode(hashlib.sha1(data).digest())[:51].decode("utf-8").lower()


@functools.lru_cache(None)
def _get_code_object_cache_dir(filename: str, firstlineno: int, name: str) -> str:
    from torch._inductor.runtime.runtime_utils import cache_dir

    # TODO: this scheme makes manual inspection of cache entries difficult,
    # consider adding some breadcrumbs in the name for ease of use
    r = os.path.join(
        cache_dir(), "pgo", sha1_hash(f"{filename}:{firstlineno}:{name}".encode("utf-8"))
    )

    log.debug(
        "get_code_object_cache_dir %s %s %s = %s", filename, firstlineno, name, r
    )
    return r


def get_code_object_cache_dir(tx: InstructionTranslator) -> str:
    return _get_code_object_cache_dir(
        tx.f_code.co_filename, tx.f_code.co_firstlineno, tx.f_code.co_name
    )


# NB: Mutable!
# TODO: probably also implement __ior__, should be marginally faster
@dataclass
class CodeObjectCache:
    automatic_dynamic: Dict[str, FrameStateSizeEntry] = field(default_factory=dict)

    def __ior__(self, other: Self) -> Self:
        inplace_merge_automatic_dynamic(self.automatic_dynamic, other.automatic_dynamic)
        return self


# TODO: move this closer to FrameStateSizeEntry
def inplace_merge_automatic_dynamic(x: Dict[str, FrameStateSizeEntry], y: Dict[str, FrameStateSizeEntry]) -> None:
    for k, v in y.items():
        if k not in x:
            x[k] = v
        else:
            x[k] |= v


@functools.lru_cache(None)
def _get_code_object_cache(
    filename: str, firstlineno: int, name: str
) -> Tuple[CodeObjectCache, Set[str]]:
    d = _get_code_object_cache_dir(filename, firstlineno, name)
    paths = os.listdir(d)
    res = CodeObjectCache()
    hashes = set()
    for path in paths:
        d_path = os.path.join(d, path)
        try:
            with open(d_path, "rb") as f:
                r = pickle.load()
                log.debug("_get_code_object_cache %s => %s", d_path, r)
                res = res.merge(r)
                hashes.add(path)
        except Exception:
            log.warning("_get_code_object_cache failed reading %s", d_path, exc_info=True)

    return res, hashes


def get_code_object_cache(
    tx: InstructionTranslator,
) -> Tuple[CodeObjectCache, Set[str]]:
    return _get_code_object_cache(
        tx.f_code.co_filename, tx.f_code.co_firstlineno, tx.f_code.co_name
    )


def get_automatic_dynamic_initial(
    tx: InstructionTranslator, name: str
) -> Optional[FrameStateSizeEntry]:
    if cache := get_code_object_cache(tx):
        r = cache.automatic_dynamic.get(name)
        if r is not None:
            log.debug("get_automatic_dynamic_initial_frame_state %s = %s", name, r)
        return r
    return None


def put_code_object_cache(
    tx: InstructionTranslator
) -> None:
    if get_code_object_cache(tx) is not None:
        return

    path = get_code_object_cache_path(tx)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        log.info("put_code_object_cache len(automatic_dynamic)=%s", len(automatic_dynamic))
        log.debug("put_code_object_cache %s", automatic_dynamic)
        pickle.dump(automatic_dynamic, f)
