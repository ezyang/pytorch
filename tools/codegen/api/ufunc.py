from tools.codegen.model import (Argument, BaseTy, BaseType, ListType,
                                 NativeFunctionsGroup, OptionalType,
                                 SelfArgument, TensorOptionsArguments, Type,
                                 assert_never)

from tools.codegen.api.types import (ArgName, BaseCType, Binding, ArrayRefCType,
                                     ConstRefCType, OptionalCType, NamedCType,
                                     tensorT, scalarT, intArrayRefT, dimnameListT,
                                     optionalTensorRefT, optionalScalarRefT, CType,
                                     BaseCppType)

from tools.codegen.api import cpp
from tools.codegen.utils import mapMaybe

from dataclasses import dataclass
from typing import Union, List, Optional

def kernel_name(g: NativeFunctionsGroup) -> str:
    return f"ufunc_{g.functional.func.name.name}"

# Tensors are omitted (as they are stored in TensorIterator), everything else is
# passed along  (technically, we can pass tensors along too, it just wastes
# argument registers)
#
# NB: used for CPU only
def dispatchstub_type(t: Type, *, binds: ArgName) -> Optional[NamedCType]:
    r = cpp.valuetype_type(t, binds=binds)
    if r is not None:
        return r

    if t == BaseTy(BaseTy.Scalar):
        return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
    elif t == BaseType(BaseTy.Tensor):
        return None
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")

def opmath_type(scalar_t: BaseCppType) -> BaseCppType:
    raise NotImplementedError

# NB: Tensors in constructor are stored in opmath_t, not scalar_t
# because Tensor in constructor = its a scalar tensor partially applied =
# it can be higher precision and we want to compute in that higher precision
#
# NB: CUDA only
def ufunctor_ctor_type(t: Type, *, binds: ArgName, scalar_t: BaseCppType) -> NamedCType:
    r = cpp.valuetype_type(t, binds=binds)
    if r is not None:
        return r

    if t == BaseTy(BaseTy.Scalar):
        return NamedCType(binds, ConstRefCType(BaseCType(scalar_t)))
    elif t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, BaseCType(opmath_type(scalar_t)))
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")

# Only Tensors ever get passed directly to operator()
#
# NB: CUDA only
def ufunctor_apply_type(t: Type, *, binds: ArgName, scalar_t: BaseCppType) -> NamedCType:
    if t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, BaseCType(scalar_t))
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")

# The actual ufunc template function the user writes.  Everything here
# is done in the computation type.  compute_t is opmath_t in CUDA and scalar_t
# in CPU
def ufunc_type(t: Type, *, binds: ArgName, compute_t: BaseCppType) -> NamedCType:
    r = cpp.valuetype_type(t, binds=binds)
    if r is not None:
        return r

    if t == BaseTy(BaseTy.Scalar):
        return NamedCType(binds, BaseCType(compute_t))
    elif t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, BaseCType(compute_t))
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")

def ufunctor_ctor_argument(a: Argument, scalar_t: BaseCppType) -> Binding:
    return Binding(
        nctype=ufunctor_ctor_type(a.type, binds=a.name, scalar_t=scalar_t),
        name=a.name,
        default=None,
        argument=a,
    )

def ufunctor_apply_argument(a: Argument, scalar_t: BaseCppType) -> Binding:
    return Binding(
        nctype=ufunctor_apply_type(a.type, binds=a.name, scalar_t=scalar_t),
        name=a.name,
        default=None,
        argument=a,
    )

def ufunc_argument(a: Argument, compute_t: BaseCppType) -> Binding:
    return Binding(
        nctype=ufunc_type(a.type, binds=a.name, compute_t=compute_t),
        name=a.name,
        default=None,
        argument=a,
    )

@dataclass(frozen=True)
class UfunctorBindings:
    ctor: List[Binding]
    apply: List[Binding]

def ufunctor_arguments(g: NativeFunctionsGroup, *, scalar_tensor: Optional[int], scalar_t: BaseCppType) -> UfunctorBindings:
    ctor = []
    apply = []
    for a in g.functional.func.arguments.flat_non_out:
        if a.type.is_tensor_like():
            if scalar_tensor == 0:
                # put it in the ctor anyway
                ctor.append(ufunctor_ctor_argument(a, scalar_t=scalar_t))
                scalar_tensor = None
            else:
                if scalar_tensor is not None:
                    scalar_tensor -= 1
                apply.append(ufunctor_apply_argument(a, scalar_t=scalar_t))
        else:
            ufunctor_ctor_argument(a, scalar_t=scalar_t)
    assert scalar_tensor is None
    return UfunctorBindings(ctor=ctor, apply=apply)

def ufunc_arguments(g: NativeFunctionsGroup, *, compute_t: BaseCppType) -> List[Binding]:
    return [ufunc_argument(a, compute_t=compute_t) for a in g.functional.func.arguments.flat_non_out]
