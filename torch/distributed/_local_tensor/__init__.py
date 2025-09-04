"""
A LocalTensor is a tensor subclass which simulates a tensor that is
distributed across SPMD ranks.  A LocalTensor might be size N, but in fact
there are world_size shards/replicas of it stored internally.  When you do a
plain PyTorch operation on it, we apply the operation to each shard; when you
do a collective, we the mathematically equivalent operation on the local
shards.  A LocalTensor is associated with a list of ranks which specify
which ranks it holds local tensors for.

NB, this is NOT a DataParallel like abstraction where you can run operations
on multiple different GPUs. It is intended purely for *debugging* purposes,
the overhead is almost certainly too high to keep eight GPUs (even the C++
autograd needs multithreading to keep up!)  (It might potentially be possible
to trace through this with torch.compile and then compile it with CUDA graphs
but this is currently a non-goal.)

We do not directly handle MPMD, as in this regime you would have to be able to
have predicated operators which only apply on a subset of ranks.  Instead, you
should generate multiple LocalTensors with disjoint lists of ranks.  However,
in this model, it still seems unavoidable that you would have to modify your
MPMD code to run all of the stages so that you actually have all the compute
for all the stages.

NB: This is a torch dispatch Tensor subclass, as we want to assume that autograd
is SPMD, so we run it once, and dispatch the inner autograd calls to the individual
local shards.

NOTE ABOUT MESH:  This subclass requires collectives that are issued to it to
respect a DeviceMesh like abstraction.  The reason for this is that when
DTensor issues us a collective for a particular rank, you will be asked to do
this on a specific process group which involves some ranks.  However, this
will only be for the LOCAL PG that this particular rank is participating in;
there will be a bunch of other PGs for other nodes that you don't get to see.
We need to be able to reverse engineer all of the collectives that don't
involve the current local rank here to actually issue them.  This can be done
two ways: (1) looking at the participating local ranks in the PG and computing
the complement which specifies all the other collectives you have to run, or
(2) retrieving the device mesh axis corresponding to the PG for this rank, and
then running all the fibers for this.

Our plan is to assume that the original list of ranks is arange and reverse engineer
the Layout corresponding to the participating ranks, with respect to the global
world size.
"""

from typing import Sequence

import torch
from torch import Tensor
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")

# Monkey patch distributed functions to detect LocalTensor usage
_original_all_reduce = None
_original_broadcast = None
_original_all_gather = None


def _has_local_tensor(*args):
    """Check if any arguments contain LocalTensor."""
    for arg in args:
        if isinstance(arg, LocalTensor):
            return True
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, LocalTensor):
                    return True
    return False


def _local_all_reduce(tensor, op=None, group=None, async_op=False):
    """Implement all_reduce for LocalTensor by applying the reduction operation across all local tensors."""
    import torch.distributed as dist

    if async_op:
        raise NotImplementedError(
            "async_op=True is not supported for LocalTensor collectives"
        )

    # Get reduction operation
    if op is None:
        op = dist.ReduceOp.SUM

    # Apply reduction across all local tensors
    reduced_tensors = {}
    all_local_tensors = list(tensor._local_tensors.values())

    for rank in tensor._local_tensors:
        if op == dist.ReduceOp.SUM:
            reduced_tensors[rank] = torch.stack(all_local_tensors).sum(dim=0)
        elif op == dist.ReduceOp.PRODUCT:
            result = all_local_tensors[0].clone()
            for t in all_local_tensors[1:]:
                result = result * t
            reduced_tensors[rank] = result
        elif op == dist.ReduceOp.MIN:
            reduced_tensors[rank] = torch.stack(all_local_tensors).min(dim=0)[0]
        elif op == dist.ReduceOp.MAX:
            reduced_tensors[rank] = torch.stack(all_local_tensors).max(dim=0)[0]
        else:
            raise ValueError(f"Unsupported reduction operation: {op}")

    # Update tensor in place (all_reduce is in-place)
    tensor._local_tensors = reduced_tensors
    return None


def _local_broadcast(tensor, src, group=None, async_op=False):
    """Implement broadcast for LocalTensor by copying src rank's tensor to all ranks."""
    if async_op:
        raise NotImplementedError(
            "async_op=True is not supported for LocalTensor collectives"
        )

    # Get the source tensor
    if src not in tensor._local_tensors:
        raise ValueError(
            f"Source rank {src} not found in LocalTensor ranks {list(tensor._local_tensors.keys())}"
        )

    src_tensor = tensor._local_tensors[src]

    # Broadcast to all ranks (copy src tensor to all local tensors)
    for rank in tensor._local_tensors:
        tensor._local_tensors[rank] = src_tensor.clone()

    return None


def _local_all_gather(tensor_list, tensor, group=None, async_op=False):
    """Implement all_gather for LocalTensor by gathering all local tensors from each rank."""
    if async_op:
        raise NotImplementedError(
            "async_op=True is not supported for LocalTensor collectives"
        )

    if not isinstance(tensor, LocalTensor):
        raise ValueError("Input tensor must be LocalTensor for LocalTensor all_gather")

    # Gather all tensors from each rank
    gathered_tensors = []
    for rank in sorted(tensor._local_tensors.keys()):
        gathered_tensors.append(tensor._local_tensors[rank])

    # Fill the tensor_list with gathered tensors
    for i, gathered_tensor in enumerate(gathered_tensors):
        if i < len(tensor_list):
            if isinstance(tensor_list[i], LocalTensor):
                # If output is LocalTensor, fill all its local tensors with the gathered tensor
                for rank in tensor_list[i]._local_tensors:
                    tensor_list[i]._local_tensors[rank] = gathered_tensor.clone()
            else:
                # If output is regular tensor, copy directly
                tensor_list[i].copy_(gathered_tensor)

    return None


class LocalTensor(torch.Tensor):
    # Map from global rank to the local tensor
    # This is a dict because in an MPMD situation, you will only have a subset
    # of ranks that this tensor actually represents (the other ranks may be
    # doing something different)
    _local_tensors: dict[int, torch.Tensor]
    _ranks: frozenset[int]  # precomputed for speed
    __slots__ = ["_local_tensors"]

    @staticmethod
    @torch._disable_dynamo
    def __new__(
        cls,
        local_tensors: dict[int, torch.Tensor],
    ) -> "LocalTensor":
        """
        .. note:: This is not a public API and it's only supposed to be used by the
            operator implementations and internals.

            TODO: guidance on alternate implementations
        """
        if any(t.requires_grad for t in local_tensors.values()):
            raise AssertionError(
                "Internal local_tensors require grad, but we will ignore those autograd graph. "
                "Make a custom autograd function and make sure you detach the inner tensors."
            )

        # Assert that all tensors are consistently the same.  I think maybe we
        # could handle size imbalance but you need to prevent code OUTSIDE of
        # LocalTensor from branching on these differences (and in general you
        # have a problem which is what size do you advertise?  It needs to be
        # some sort of unbacked-like thing).

        it = iter(local_tensors.items())
        first_rank, first_local_tensor = next(it)

        shape = first_local_tensor.shape
        strides = first_local_tensor.stride()
        dtype = first_local_tensor.dtype
        device = first_local_tensor.device
        layout = first_local_tensor.layout

        # TODO: make this less horrible
        def get_extra_dispatch_keys(t):
            extra_dispatch_keys = torch._C.DispatchKeySet.from_raw_repr(0)
            if torch._C._dispatch_keys(t).has(torch._C.DispatchKey.Conjugate):
                extra_dispatch_keys = extra_dispatch_keys.add(
                    torch._C.DispatchKey.Conjugate
                )
            if torch._C._dispatch_keys(t).has(torch._C.DispatchKey.Negative):
                extra_dispatch_keys = extra_dispatch_keys.add(
                    torch._C.DispatchKey.Negative
                )
            return extra_dispatch_keys

        extra_dispatch_keys = get_extra_dispatch_keys(first_local_tensor)

        for _rank, local_tensor in it:
            assert shape == local_tensor.shape
            assert strides == local_tensor.stride()
            assert dtype == local_tensor.dtype
            assert layout == local_tensor.layout
            assert extra_dispatch_keys == get_extra_dispatch_keys(local_tensor)

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=strides,
            dtype=dtype,
            device=device,
            layout=layout,
            requires_grad=False,
            _extra_dispatch_keys=extra_dispatch_keys,
        )

        r._local_tensors = local_tensors
        r._ranks = frozenset(local_tensors.keys())
        return r

    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __repr__(self):
        return f"LocalTensor(local_tensors={self._local_tensors})"

    def __tensor_flatten__(self):
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
        return ["_local_tensors"], ()

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        assert flatten_spec is not None, (
            "Expecting spec to be not None from `__tensor_flatten__` return value!"
        )
        local_tensors = inner_tensors["_local_tensors"]
        return LocalTensor(local_tensors)

    @classmethod
    @torch._disable_dynamo
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # This is horribly inefficient
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        local_tensor = None
        for arg in flat_args:
            if isinstance(arg, LocalTensor):
                local_tensor = arg
                break

        assert local_tensor is not None

        with LocalTensorMode(local_tensor._ranks):
            return func(*args, **kwargs)


def _check_for_subclass(flat_args: Sequence[object]) -> bool:
    return any(_check_for_subclass_arg(x) for x in flat_args)


def _check_for_subclass_arg(x: object) -> bool:
    return (
        not isinstance(x, LocalTensor)
        and isinstance(x, Tensor)
        and type(x) is not Tensor
        and type(x) is not torch.nn.Parameter
    )


class LocalTensorMode(TorchDispatchMode):
    # What ranks this local tensor mode is operating over
    def __init__(self, ranks: frozenset[int] = None):
        self.ranks = ranks

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        flat_args, args_spec = pytree.tree_flatten((args, kwargs))

        # Find all LocalTensor arguments to determine ranks
        local_tensors = [a for a in flat_args if isinstance(a, LocalTensor)]

        if not local_tensors:
            # No LocalTensors involved, pass through to regular dispatch
            return func(*args, **kwargs)

        # Use ranks from the first LocalTensor if not specified
        if self.ranks is None:
            self.ranks = local_tensors[0]._ranks

        # Check for unrecognized tensor subclasses (but allow regular tensors and scalars)
        has_unrecognized_types = _check_for_subclass(flat_args)
        if has_unrecognized_types:
            unrecognized_types = [
                type(x) for x in flat_args if _check_for_subclass_arg(x)
            ]
            not_implemented_log.debug(
                "LocalTensorMode unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        # For LocalTensors, verify they have compatible ranks
        for a in flat_args:
            if isinstance(a, LocalTensor):
                # For different rank sets, use intersection
                if a._ranks != self.ranks:
                    self.ranks = self.ranks & a._ranks
                    if not self.ranks:
                        raise ValueError("No common ranks between LocalTensors")

        flat_rank_rets = {}
        for r in sorted(self.ranks):
            rank_flat_args = [
                a._local_tensors[r] if isinstance(a, LocalTensor) else a
                for a in flat_args
            ]
            rank_args, rank_kwargs = pytree.tree_unflatten(rank_flat_args, args_spec)
            rank_ret = func(*rank_args, **rank_kwargs)
            flat_rank_rets[r] = pytree.tree_flatten(rank_ret)[0]

        # Create the LocalTensors
        rr_key = next(iter(flat_rank_rets.keys()))
        rr_val = flat_rank_rets[rr_key]

        if isinstance(rr_val, Tensor):
            return LocalTensor({r: flat_rank_rets[r] for r in sorted(self.ranks)})

        if isinstance(rr_val, (list, tuple)):
            ret = []
            for i in range(len(rr_val)):
                rets = {r: flat_rank_rets[r][i] for r in sorted(self.ranks)}
                v_it = iter(rets.values())
                v = next(v_it)
                if isinstance(v, Tensor):
                    ret.append(LocalTensor(rets))
                else:
                    assert all(v == v2 for v2 in v_it)
                    ret.append(v)

            if len(ret) == 1:
                return ret[0]
            return tuple(ret)
        else:
            # Single non-tensor return value (scalar, etc.)
            v_it = iter(flat_rank_rets.values())
            v = next(v_it)
            assert all(v == v2 for v2 in v_it)
            return v
