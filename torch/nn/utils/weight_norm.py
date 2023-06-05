r"""
Weight Normalization from https://arxiv.org/abs/1602.07868
"""
from torch.nn.parameter import Parameter, UninitializedParameter
from torch import _weight_norm, norm_except_dim
import torch.utils._copy
from typing import Any, TypeVar
from ..modules import Module
import types

__all__ = ['WeightNorm', 'weight_norm', 'remove_weight_norm']

class WeightNorm:
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    # TODO Make return type more specific
    def compute_weight(self, module: Module) -> Any:
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        return _weight_norm(v, g, self.dim)

    @staticmethod
    def apply(module, name: str, dim: int) -> 'WeightNorm':
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError("Cannot register two weight_norm hooks on "
                                   "the same parameter {}".format(name))

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        weight = getattr(module, name)
        if isinstance(weight, UninitializedParameter):
            raise ValueError(
                'The module passed to `WeightNorm` can\'t have uninitialized parameters. '
                'Make sure to run the dummy forward before applying weight normalization')
        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + '_g', Parameter(norm_except_dim(weight, 2, dim).data))
        module.register_parameter(name + '_v', Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        # NB: Repeated deepcopy doesn't work
        old_deepcopy = getattr(module, "__deepcopy__", torch.utils._copy.default_deepcopy)
        def new_deepcopy(self, memo):
            uncopyable_weight = getattr(module, name)
            try:
                # Skip the uncopyable Tensor...
                setattr(module, name, None)
                r = old_deepcopy(self, memo)
                # ...and recompute it the normal way
                setattr(r, name, fn.compute_weight(r))
                return r
            finally:
                setattr(module, name, uncopyable_weight)
        new_deepcopy.old_deepcopy = old_deepcopy
        setattr(module, "__deepcopy__", types.MethodType(new_deepcopy, module))

        # Assume that we're getting the nn.Module getstate which only ever
        # returns a dict
        old_getstate = getattr(module, "__getstate__")
        def new_getstate(self):
            r = old_getstate()
            assert isinstance(r, dict)
            del r['__deepcopy__']
            del r['__getstate__']
            return r
        new_getstate.old_getstate = old_getstate
        setattr(module, "__getstate__", types.MethodType(new_getstate, module))

        return fn

    def remove(self, module: Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_v']
        setattr(module, self.name, Parameter(weight.data))
        if hasattr(module, '__deepcopy__'):
            if getattr(module.__deepcopy__, 'old_deepcopy', None) is torch.utils._copy.default_deepcopy:
                delattr(module, "__deepcopy__")
            else:
                setattr(module, "__deepcopy__", module.__deepcopy__.old_deepcopy)
        if hasattr(module.__getstate__, 'old_getstate'):
            setattr(module, "__getstate__", module.__getstate__.old_getstate)

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))


T_module = TypeVar('T_module', bound=Module)

def weight_norm(module: T_module, name: str = 'weight', dim: int = 0) -> T_module:
    r"""Applies weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNorm.apply(module, name, dim)
    return module


def remove_weight_norm(module: T_module, name: str = 'weight') -> T_module:
    r"""Removes the weight normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))
