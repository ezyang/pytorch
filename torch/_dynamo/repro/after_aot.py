import argparse
import copy
import functools
import io
import itertools
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
from importlib import import_module
from tempfile import TemporaryFile
from typing import Any, Optional, Sequence

import torch
import torch._prims_common as utils
import torch.fx as fx
from torch._dynamo.debug_utils import (
    _cuda_system_info_comment,
    AccuracyError,
    backend_accuracy_fails,
    BuckTargetWriter,
    cast_to_fp64,
    extra_imports,
    generate_config_string,
    helper_for_dump_minify,
    minifier_dir,
    NNModuleToString,
    same_two_models,
    TEST_REPLACEABLE_COMMENT,
)
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import clone_inputs, counters, same
from torch.hub import tqdm
from torch.multiprocessing.reductions import StorageWeakRef

from torch.utils._content_store import ContentStoreReader, ContentStoreWriter

from .. import config

import torch._functorch.fx_minifier
torch._functorch.fx_minifier.READER = ContentStoreReader('/home/ezyang/local/a/pytorch/torch_compile_debug/run_2023_04_28_10_11_59_260195-pid_2880226/minifier/checkpoints/')

log = logging.getLogger(__name__)


inductor_config = import_module("torch._inductor.config")
use_buck = inductor_config.is_fbcode()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           MAIN ENTRY POINT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def wrap_compiler_debug(unconfigured_compiler_fn, compiler_name: str):
    """
    Minifier for Fx Graph modules after Aot Autograd has finished. We wrap both
    forward and backward call separately with the backend compiler_fn - like
    inductor or nvfuser. Intercepting after Aot Autograd presents neat
    abstraction, where all the params are lifted as graph inputs, making it easy
    to save the graph as a string.
    """

    @functools.wraps(unconfigured_compiler_fn)
    def debug_wrapper(gm, example_inputs, **kwargs):
        from torch._subclasses import FakeTensorMode

        compiler_fn = functools.partial(unconfigured_compiler_fn, **kwargs)

        from torch._functorch.aot_autograd import get_aot_graph_name

        graph_name = get_aot_graph_name()

        # TODO: why do we need to deepcopy the original graph?
        orig_graph = copy.deepcopy(gm.graph)
        assert config.repro_after in ("dynamo", "aot", None)

        try:
            # Call the compiler_fn - which is either aot_autograd or inductor
            # with fake inputs
            inner_compiled_fn = compiler_fn(gm, example_inputs)
        except Exception as e:
            # TODO: Failures here are troublesome because no real inputs,
            # need a different serialization strategy
            if config.repro_after == "aot":
                if config.repro_level == 1:
                    dump_compiler_graph_state(
                        fx.GraphModule(gm, orig_graph),
                        example_inputs,
                        compiler_name,
                    )
                elif config.repro_level == 2:
                    dump_to_minify(
                        fx.GraphModule(gm, orig_graph),
                        example_inputs,
                        compiler_name,
                    )
                log.error("CompilerError")
            raise

        # We may run regular PyTorch compute that may trigger Dynamo, do NOT
        # recursively attempt to accuracy minify in that case!
        def deferred_for_real_inputs(real_inputs):
            # This is a bit obscure: if we recursively try to accuracy minify
            # the SAME function, this would trigger.  But most of the time
            # we should never hit this branch
            if config.repro_after != "aot":
                return inner_compiled_fn(real_inputs)
            with config.patch(repro_after=None):
                return inner_debug_fn(real_inputs)

        def inner_debug_fn(real_inputs):
            """
            Aot Autograd fw_compiler and bw_compiler can have fake tensors. So,
            example_inputs can be fake tensors. We can call compiler_fn (which is
            inductor or nvfuser) with fake tensors but the actually compiled_fn
            should be called with real tensors. Therefore, the actual invocation
            is deferred.
            """
            # Copy the tensor attrs like shape, stride etc by converting to Fake Tensor
            # because inductor clears the tensor list in its codegen. And example_inputs
            # are available only for the first invocation.
            fake_mode = FakeTensorMode()
            copy_tensor_attrs = [
                fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x
                for x in real_inputs
            ]
            if config.repro_level == 3:
                # Always dump the original module in case we have segfaults
                dump_to_minify(
                    fx.GraphModule(gm, orig_graph), real_inputs, compiler_name
                )

            if config.repro_level == 4:
                if compiler_name != "inductor":
                    raise NotImplementedError(
                        "Accuracy minification is supported for inductor only"
                    )
                if backend_aot_accuracy_fails(gm, real_inputs, compiler_fn):
                    log.warning(
                        "Accuracy failed for the AOT Autograd graph %s", graph_name
                    )
                    dump_compiler_graph_state(
                        fx.GraphModule(gm, orig_graph),
                        real_inputs,
                        f"{compiler_name}_accuracy",
                    )
                    dump_to_minify(
                        fx.GraphModule(gm, orig_graph),
                        real_inputs,
                        f"{compiler_name}_accuracy",
                    )
                    raise AccuracyError("Bad accuracy detected")
                else:
                    # Call the compiled function with real inputs
                    return inner_compiled_fn(real_inputs)
            else:
                try:
                    # Call the compiled function with real inputs
                    out = inner_compiled_fn(real_inputs)
                    # sync cuda kernels to ensure IMA detection
                    for arg in example_inputs:
                        if isinstance(arg, torch.Tensor) and arg.is_cuda:
                            torch.cuda.synchronize()
                            break
                    return out
                except Exception as e:
                    if config.repro_level == 1:
                        dump_compiler_graph_state(
                            fx.GraphModule(gm, orig_graph),
                            copy_tensor_attrs,
                            compiler_name,
                        )
                    elif config.repro_level == 2:
                        dump_to_minify(
                            fx.GraphModule(gm, orig_graph),
                            copy_tensor_attrs,
                            compiler_name,
                        )
                    raise

        if config.repro_after == "aot":
            compiled_fn = deferred_for_real_inputs
            compiled_fn._boxed_call = True  # type: ignore[attr-defined]
            return compiled_fn
        else:
            return inner_compiled_fn

    return debug_wrapper


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       REPRO SUPPORT CODE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# Helper functions for computing what the default values of tensor
# values should be.  These all coincide with factory functions, e.g., torch.empty


def _stride_or_default(
    stride: Optional[Sequence[int]], *, shape: Sequence[int]
) -> Sequence[int]:
    return stride if stride is not None else utils.make_contiguous_strides_for(shape)


def _dtype_or_default(dtype: Optional[torch.dtype]) -> torch.dtype:
    return dtype if dtype is not None else torch.float32


def _device_or_default(device: Optional[torch.device]) -> torch.device:
    return device if device is not None else torch.device("cpu")


def _storage_offset_or_default(storage_offset: Optional[int]) -> int:
    return storage_offset if storage_offset is not None else 0


# TODO: Support bundling the entire repro into a zip file for ease of
# transferring around
class InputReader:
    def __init__(self, save_dir=None, total=None):
        # If None, we will generate random data instead.  It's important
        # to natively support this use case as it will allow people to
        # share repros without including the real data, if the problem
        # reproduces even on random data.
        if save_dir is None:
            log.warning("no save_dir specified, will generate random data")
        self.store = ContentStoreReader(save_dir) if save_dir is not None else None
        self.total = total
        self.pbar = tqdm(desc="Loading inputs", total=total)

    def storage(self, storage_hash, nbytes, *, device=None, dtype_hint=None):
        self.pbar.update(1)
        if self.pbar.n >= self.total:
            self.pbar.close()
        device = _device_or_default(device)
        dtype_hint = _dtype_or_default(dtype_hint)
        if self.store is not None and storage_hash is not None:
            try:
                storage = self.store.read_storage(storage_hash)
            except FileNotFoundError:
                pass
            else:
                if device != storage.device:
                    log.warning("device mismatch: %s != %s", device, storage.device)
                    # TODO: transfer it to the right device?  But failing this
                    # way would be very mysterious!  Would have been better
                    # not to store device in the serialized format...
                return storage
        log.warning("could not load %s, generating random data instead", storage_hash)
        shape = (nbytes // dtype_hint.itemsize,)
        stride = _stride_or_default(None, shape=shape)
        return rand_strided(shape, stride, dtype_hint, device).untyped_storage()

    def tensor(
        self,
        storage,
        shape,
        stride=None,
        *,
        storage_offset=None,
        dtype=None,
        **metadata,
    ):
        self.pbar.update(1)
        stride = _stride_or_default(stride, shape=shape)
        storage_offset = _storage_offset_or_default(storage_offset)
        dtype = _dtype_or_default(dtype)
        t = torch.tensor([], dtype=dtype, device=storage.device)
        t.set_(storage, storage_offset, shape, stride)
        torch._utils.set_tensor_metadata(t, metadata)
        return t

    def symint(self, val):
        self.pbar.update(1)
        return val


# Here is our writer strategy:
#  1. We will stream all of the inputs to disk
#  2. You can now deterministically randomize the inputs, or reload
#     the inputs from disk
#  3. You can YOLO run the script without the inputs, in which case
#     we'll fill the inputs with random data and pray.  This is the
#     legacy behavior, but it's also useful if you want to find out
#     if we're so broken even random inputs trigger it
#  4. We could offer an in process "check if the randomized thing
#     works too" but this is delicate so we don't do it


class InputWriter:
    def __init__(self, save_dir, *, stable_hash=False):
        self._lines = []
        # TODO: consider ensuring tensor and storage counters line up?
        self.tensor_counter = itertools.count()
        self.symint_counter = itertools.count()
        self.storage_counter = itertools.count()
        self.save_dir = save_dir
        self.store = (
            ContentStoreWriter(save_dir, stable_hash=stable_hash)
            if save_dir is not None
            else None
        )
        self.seen_storages = {}

    def lines(self):
        r = [
            "import torch._dynamo.repro.after_aot",
            f"reader = torch._dynamo.repro.after_aot.InputReader(save_dir={self.save_dir!r}, total={len(self._lines)})",
        ]
        r.extend(self._lines)
        return r

    # Storages are untyped, but we need to initialize them with data if
    # we don't have the real data, so we give a hint saying what kind
    # of initialization may be appropriate
    #
    # If we had a FakeTensor, device_hint tells us what device should be
    def storage(self, untyped_storage, *, dtype_hint=None, device_hint=None) -> str:
        ws = StorageWeakRef(untyped_storage)
        v = self.seen_storages.get(ws)
        if v is not None:
            return v
        v = f"buf{next(self.storage_counter)}"
        maybe_dtype_hint = ""
        if _dtype_or_default(None) != _dtype_or_default(dtype_hint):
            maybe_dtype_hint = f", dtype_hint={dtype_hint!r}"
        # TODO: being optional on device is kind of pointless as the default
        # is CPU but most repros we care about are CUDA
        maybe_device = ""
        device = untyped_storage.device
        if device.type == "meta":
            assert device_hint is not None
            device = device_hint
        if _device_or_default(None) != device:
            maybe_device = f", device={device!r}"
        nbytes = untyped_storage.nbytes()
        storage_hash = None
        if self.store is not None and untyped_storage.device.type != "meta":
            storage_hash = self.store.write_storage(untyped_storage)
        self._lines.append(
            f"{v} = reader.storage({storage_hash!r}, {nbytes!r}{maybe_device}{maybe_dtype_hint})"
        )
        self.seen_storages[ws] = v
        return v

    def tensor(self, t) -> str:
        storage = self.storage(
            t.untyped_storage(), dtype_hint=t.dtype, device_hint=t.device
        )
        maybe_stride = ""
        if _stride_or_default(None, shape=t.shape) != t.stride():
            maybe_stride = f", {tuple(t.stride())}"
        maybe_dtype = ""
        if _dtype_or_default(None) != t.dtype:
            maybe_dtype = f", dtype={t.dtype!r}"
        maybe_storage_offset = ""
        if _storage_offset_or_default(None) != t.storage_offset():
            maybe_storage_offset = f", storage_offset={t.storage_offset()!r}"
        maybe_tensor_metadata = ""
        tensor_metadata = torch._utils.get_tensor_metadata(t)
        if tensor_metadata:
            maybe_tensor_metadata = ", " + ", ".join(
                f"{k}={v!r}" for k, v in tensor_metadata.items()
            )
        v = f"t{next(self.tensor_counter)}"
        self._lines.append(
            f"{v} = reader.tensor({storage}, {tuple(t.shape)}"
            f"{maybe_stride}{maybe_storage_offset}{maybe_dtype}{maybe_tensor_metadata})"
        )
        return v

    # TODO: this doesn't actually symint atm
    def symint(self, val) -> str:
        if isinstance(val, torch.SymInt):
            val = val.node.hint
        v = f"s{next(self.symint_counter)}"
        self._lines.append(f"{v} = reader.symint({val!r})")
        return v


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           DUMP REPROS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def generate_compiler_repro_string(gm, args, *, stable_output=False, save_dir=None):
    model_str = textwrap.dedent(
        f"""
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

{generate_config_string(stable_output=stable_output)}

isolate_fails_code_str = None

{TEST_REPLACEABLE_COMMENT}
{extra_imports}

        """
    )
    if not stable_output:
        model_str += f"# torch version: {torch.version.__version__}\n"
        if hasattr(torch.version, "cuda"):
            model_str += f"# torch cuda version: {torch.version.cuda}\n"
        if hasattr(torch.version, "git_version"):
            model_str += f"# torch git version: {torch.version.git_version}\n\n\n"
        model_str += _cuda_system_info_comment()

    model_str += NNModuleToString.convert(gm)

    # get hint shape/stride when dynamic shape enabled
    def hint_if_symint(x):
        return tuple(i.node.hint if isinstance(i, torch.SymInt) else i for i in x)

    writer = InputWriter(save_dir)
    wargs = []
    for i, arg in enumerate(args):
        if isinstance(arg, (int, torch.SymInt)):
            wargs.append(writer.symint(arg))
        elif isinstance(arg, torch.Tensor):
            # TODO: improve these names with FQN
            wargs.append(writer.tensor(arg))
        else:
            raise TypeError(f"arg is neither SymInt/int nor torch.Tensor, {arg}")

    model_str += "\n".join(writer.lines()) + "\n"
    model_str += f"args = [{', '.join(wargs)}]\n"

    # TODO: fake may be better for performance here
    tracing_mode = "real"
    if config.dynamic_shapes:
        tracing_mode = "symbolic"
    model_str += f"mod = make_fx(Repro(), tracing_mode={repr(tracing_mode)})(*args)\n"
    return model_str


def save_graph_repro(
    fd, gm, args, compiler_name, *, stable_output=False, save_dir=None, minify=False
):
    # TODO: not sure why we need this import
    if "inductor" in compiler_name:
        fd.write("import torch._inductor.overrides\n")
    fd.write(
        generate_compiler_repro_string(
            gm, args, stable_output=stable_output, save_dir=save_dir
        )
    )
    accuracy = "_accuracy" in compiler_name
    fd.write("if __name__ == '__main__':\n")
    fd.write("    from torch._dynamo.repro.after_aot import run_repro\n")
    fd.write(
        f"    run_repro(mod, args, accuracy={accuracy!r}, minify={minify!r}, "
        f"save_dir={save_dir!r}, patch_code=isolate_fails_code_str)\n"
    )


def dump_compiler_graph_state(gm, args, compiler_name, *, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(minifier_dir(), "checkpoints")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f"{len(gm.graph.nodes)}.py")
    if os.path.exists(file_name):
        # TODO: Don't overwrite the most RECENT checkpoint
        log.warning("Overwriting preexisting checkpoint at %s", file_name)
        os.rename(file_name, f"{file_name}.bak")
    log.warning(
        "Writing checkpoint with %s nodes to %s", len(gm.graph.nodes), file_name
    )
    with open(file_name, "w") as fd:
        save_graph_repro(fd, gm, args, compiler_name, save_dir=save_dir)
    curdir = os.getcwd()
    repro_path = os.path.join(curdir, "repro.py")
    try:
        shutil.copyfile(file_name, repro_path)
        log.warning("Copying repro file for convenience to %s", repro_path)
        if use_buck:
            BuckTargetWriter(file_name).write()
    except OSError:
        log.warning("No write permissions for %s", repro_path)
        pass


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           DUMP MINIFIER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def dump_to_minify(gm, args, compiler_name: str):
    out = io.StringIO()
    # TODO: factor this out
    subdir = os.path.join(minifier_dir(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    save_graph_repro(out, gm, args, compiler_name, save_dir=subdir, minify=True)
    return helper_for_dump_minify(out.getvalue())


def isolate_fails(
    fail_fn, fx_g, args, compiler_name: str, env=None, patch_code=None, save_dir=None
):
    if env is None:
        env = {}
    subdir = os.path.join(os.getcwd(), "isolate")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{str(uuid.uuid4())[:5]}.py")
    with open(file_name, "w") as fd:
        fd.write("import torch._functorch.fx_minifier\n")  # Hack before relocating custom op
        fd.write("import torch.utils._content_store\n")  # Hack before relocating custom op
        fd.write(f"torch._functorch.fx_minifier.READER = torch.utils._content_store.ContentStoreReader({save_dir!r})\n")  # Hack before relocating custom op
        repro_code = generate_compiler_repro_string(fx_g, args, save_dir=save_dir)
        if patch_code is not None:
            repro_code = repro_code.replace(TEST_REPLACEABLE_COMMENT, patch_code)
        fd.write(repro_code)
        fd.write(
            textwrap.dedent(
                f"""
                from torch._dynamo.repro.after_aot import {fail_fn}
                """
            )
        )
        fd.write(
            textwrap.dedent(
                f"""
                if {fail_fn}(mod, args):
                    exit(1)
                else:
                    exit(0)
                """
            )
        )
    # with open(file_name, "r") as fd:
    #     print(fd.read())
    new_env = os.environ.copy()
    new_env = {**new_env, **env}
    stdout, stderr = TemporaryFile(), TemporaryFile()

    if use_buck:
        cmd = BuckTargetWriter(file_name).write(print_msg=False)
    else:
        cmd = ["python", file_name]

    p = subprocess.Popen(
        cmd,
        cwd=subdir,
        stdout=stdout,
        stderr=stderr,
        env=new_env,
    )
    p.wait()

    stdout.seek(0)
    stderr.seek(0)
    print(textwrap.indent(stdout.read().decode("utf-8"), prefix=">>  "))
    print(textwrap.indent(stderr.read().decode("utf-8"), prefix=">>  "))
    # print(f"Isolated test failed - {file_name}")
    if p.returncode != 0:
        return True
    return False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       MINIFIER TOOLS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def inductor_fails(fx_g, args, check_str=None):
    has_cuda = False
    for arg in args:
        if arg.is_cuda:
            has_cuda = True
            break

    def sync():
        if has_cuda:
            # Ensures that segfaults are surfaced
            torch.cuda.synchronize()

    from torch._inductor.compile_fx import compile_fx_inner

    try:
        result = fx_g(*args)
        assert isinstance(result, (tuple, list))
        assert not any(isinstance(x, (tuple, list)) for x in result)
    except Exception:
        return False

    sync()

    try:
        compile_mod = compile_fx_inner(fx_g, args)
        compile_mod(args)
        sync()
    except Exception as e:
        if check_str is not None and check_str not in repr(e):
            return False
        print(repr(e))
        return True
    return False


def inductor_accuracy_fails(fx_g, args, check_str=None):
    from torch._inductor.compile_fx import compile_fx_inner

    return backend_aot_accuracy_fails(fx_g, args, compile_fx_inner)


backend_aot_accuracy_fails = functools.partial(backend_accuracy_fails, only_fwd=True, require_fp64=True)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           REPRO MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def repro_main(options, mod, args):
    from torch._inductor.compile_fx import compile_fx_inner
    import torch._functorch.fx_minifier as fx_minifier

    fx_minifier.READER = ContentStoreReader(options.save_dir)

    # Invariant for graphs we generate with the repro script
    assert not any(mod.named_parameters()) and not any(mod.named_buffers())

    compiler_name = "inductor_accuracy" if options.accuracy else "inductor"
    torch._inductor.config.generate_intermediate_hooks = True

    if options.minify:
        favored_device = 1 if torch.cuda.device_count() >= 2 else 0
        env_variables = {"CUDA_VISIBLE_DEVICES": str(favored_device)}

        module_fails: Any
        if options.isolate:
            module_fails = functools.partial(
                isolate_fails,
                "inductor_accuracy_fails" if options.accuracy else "inductor_fails",
                env=env_variables,
                compiler_name=compiler_name,
                patch_code=options.patch_code,
                save_dir=options.save_dir,
            )
        else:
            module_fails = (
                inductor_accuracy_fails if options.accuracy else inductor_fails
            )

        fx_minifier.minifier(
            mod,
            args,
            module_fails=module_fails,
            dump_state=functools.partial(
                dump_compiler_graph_state,
                compiler_name=compiler_name,
                save_dir=options.save_dir if options.inplace else None,
            ),
            save_dir=options.save_dir,
            offload_to_disk=options.offload_to_disk,
            skip_offload=options.skip_saving_eager_intermediates,
            skip_sanity=options.skip_sanity,
            max_granularity=options.max_granularity,
        )

    elif options.analyze:
        # TODO: The logic for cloning inputs/models here is intentionally
        # modeled off of run_fwd_maybe_bwd, but arguably it is better not to
        # clone inputs (as you are doubling your effective GPU memory usage).
        # It is certainly faster though!  It probably makes sense to let the
        # user specify the offload strategy.
        from torch._inductor.hooks import intermediate_hook

        with tqdm(desc="Compiling"):
            compiled = compile_fx_inner(mod, args)
        total = counters["inductor"]["intermediate_hooks"]

        known_names = set()

        def save_hook(name, val):
            known_names.add(name)
            if not options.skip_saving_inductor_intermediates:
                writer.write_tensor(os.path.join("inductor", name), val)
            pbar.update(1)

        writer = torch.utils._content_store.ContentStoreWriter(
            options.save_dir, stable_hash=options.stable_hash
        )
        reader = torch.utils._content_store.ContentStoreReader(options.save_dir)

        new_args = clone_inputs(args)
        with intermediate_hook(save_hook), tqdm(
            desc="Saving inductor intermediates", total=total
        ) as pbar:
            compiled(new_args)
            assert not new_args

        def compare_tuples(tuple1, tuple2):
            diff_indices = [i for i in range(len(tuple1)) if tuple1[i] != tuple2[i]]
            diff_values = [(tuple1[i], tuple2[i]) for i in diff_indices]

            if not diff_values:
                return None
            else:
                return " and ".join(f"{a} != {b}" for a, b in diff_values)

        def check_hook(name, val):
            meta = writer.compute_tensor_metadata(val)
            meta2 = reader.read_tensor_metadata(os.path.join("inductor", name))
            reason = compare_tuples(meta, meta2)
            if reason is not None:
                pbar.write(f"NONDETERMINISTIC INDUCTOR at {name} ({reason})")
            pbar.update(1)

        if not options.skip_check_deterministic:
            new_args = clone_inputs(args)
            with intermediate_hook(check_hook), tqdm(
                desc="Checking inductor determinism", total=total
            ) as pbar:
                compiled(new_args)
                assert not new_args

        class WriterInterp(fx.Interpreter):
            def __init__(self, mod, subdir):
                super().__init__(mod)
                self.subdir = subdir

            def run_node(self, n):
                r = super().run_node(n)
                name = n.name
                if name in known_names:
                    pbar.update(1)
                    writer.write_tensor(os.path.join(self.subdir, name), r)
                return r

        # NB: the module cast doesn't actually do anything, since there are no
        # parameters/buffers on the module
        if not options.skip_saving_float64_intermediates:
            new_mod, new_args = cast_to_fp64(mod, clone_inputs(args))
            with tqdm(desc="Saving float64 intermediates", total=total) as pbar:
                WriterInterp(new_mod, "float64").boxed_run(new_args)
            assert not new_args

        class ExactReaderInterp(fx.Interpreter):
            def run_node(self, n):
                r = super().run_node(n)
                name = n.name
                if name in known_names:
                    meta = writer.compute_tensor_metadata(r)
                    meta2 = reader.read_tensor_metadata(os.path.join("float64", name))
                    reason = compare_tuples(meta, meta2)
                    if reason is not None:
                        pbar.write(f"NONDETERMINISTIC FLOAT64 at {name} ({reason})")
                    pbar.update(1)
                return r

        # TODO: check eager determinism

        if not options.skip_check_deterministic:
            new_mod, new_args = cast_to_fp64(mod, clone_inputs(args))
            with tqdm(desc="Checking float64 determinism", total=total) as pbar:
                ExactReaderInterp(new_mod).boxed_run(new_args)
                assert not new_args

        # Now that we've saved everything, interp through the eager graph
        # and do comparisons
        class ReaderInterp(fx.Interpreter):
            def run_node(self, n):
                r = super().run_node(n)
                name = n.name
                if name in known_names:
                    inductor = reader.read_tensor(os.path.join("inductor", name))
                    float64 = reader.read_tensor(os.path.join("float64", name))
                    logged = False

                    def log_error(msg, *args):
                        nonlocal logged
                        logged = True
                        pbar.write(f"DIVERGED at {name}: {msg % args}")

                    if not same(
                        r,
                        inductor,
                        float64,
                        tol=torch._dynamo.config.repro_tolerance,
                        equal_nan=True,
                        log_error=log_error,
                    ):
                        assert logged
                    pbar.update(1)
                return r

        with tqdm(desc="Checking divergence", total=total) as pbar:
            ReaderInterp(mod).boxed_run(args)
        assert not args

    else:
        from torch.cuda import synchronize

        compiled = compile_fx_inner(mod, args)

        if options.accuracy:
            if not same_two_models(mod, compiled, args, only_fwd=True):
                raise AccuracyError("Bad accuracy detected")
        else:
            need_sync = False
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.is_cuda:
                    need_sync = True
                    break
            ref = compiled(args)
            if need_sync:
                synchronize()  # ensure segfaults are surfaced


# TODO: lazily load the inputs or something, rather than cloning them
def run_repro(
    mod, args, *, accuracy=False, minify=False, save_dir=None, patch_code=None
):
    parser = argparse.ArgumentParser(description="torch.compile repro script")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--minify",
        action="store_true",
        default=minify,
        help="run the minifier on the repro",
    )
    group.add_argument(
        "--analyze", action="store_true", help="run the analyzer on the repro"
    )

    accuracy_group = parser.add_mutually_exclusive_group()
    accuracy_group.add_argument(
        "--accuracy",
        action="store_true",
        default=accuracy,
        help="test accuracy when running repro",
    )
    accuracy_group.add_argument(
        "--no-accuracy",
        dest="accuracy",
        action="store_false",
        default=accuracy,
        help="do not test accuracy",
    )

    isolate_group = parser.add_mutually_exclusive_group()
    isolate_group.add_argument(
        "--isolate",
        action="store_true",
        default=True,
        help="run all --minify in separate processes to avoid interference",
    )
    isolate_group.add_argument(
        "--no-isolate",
        dest="isolate",
        action="store_false",
        help="speed up --minify by running all compilation in same process",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=save_dir,
        help="directory where saved inputs live",
    )
    parser.add_argument(
        "--patch-code", type=str, default=patch_code, help="patch code for testing"
    )
    parser.add_argument(
        "--skip-saving-inductor-intermediates",
        action="store_true",
        help="skip saving inductor intermediates on --analyze",
    )
    parser.add_argument(
        "--skip-saving-float64-intermediates",
        action="store_true",
        help="skip saving float64 intermediates on --analyze",
    )
    parser.add_argument(
        "--skip-saving-eager-intermediates",
        action="store_true",
        help="skip saving eager intermediates on --minify",
    )
    parser.add_argument(
        "--skip-check-deterministic",
        action="store_true",
        help="skip checking that the network is deterministic on --analyze",
    )
    parser.add_argument(
        "--stable-hash",
        action="store_true",
        help="use SHA-1 checksum instead of fast (but possibly unsound) hash",
    )
    # TODO: make this control --analyze too, aka have an in-memory version of
    # analyze
    parser.add_argument(
        "--offload-to-disk",
        action="store_true",
        help="during minification, offload delta debugging intermediates to disk.  Use if you're OOMing",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="run minification in same directory as script; useful if you're resuming from checkpoint",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="skip sanity check at beginning of minification on original graph",
    )
    parser.add_argument(
        "--max-granularity",
        type=int,
        default=None,
        help="start at this granularity and work down",
    )

    options = parser.parse_args()
    repro_main(options, mod, args)
