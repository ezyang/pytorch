from tools.codegen.model import (Argument, BaseTy, BaseType, ListType,
                                 NativeFunctionsGroup, OptionalType,
                                 SelfArgument, TensorOptionsArguments, Type,
                                 assert_never)

from tools.codegen.api.types import (ArgName, BaseCType, Binding, ArrayRefCType,
                                     ConstRefCType, OptionalCType, NamedCType,
                                     tensorT, scalarT, intArrayRefT, dimnameListT,
                                     optionalTensorRefT, optionalScalarRefT, CType)

from tools.codegen.api import cpp
from tools.codegen.utils import mapMaybe

from typing import Union, List, Optional

# The set of types ufuncs support is very limited indeed >:)
def argumenttype_type(t: Type, *, binds: ArgName, refine: Optional[CType], tensor: bool) -> Optional[NamedCType]:
    if tensor:
        # We only do type translation for tensors within refined contexts (where we
        # know what the dtype is statically)
        assert refine is not None
        if t == BaseType(BaseTy.Tensor):
            return NamedCType(binds, refine)
        if t.is_tensor_like():
            raise AssertionError(f"unrecognized type {repr(t)}")
        return None
    else:
        if t.is_tensor_like():
            return None

        # If it's a value type, do the value type translation
        r = cpp.valuetype_type(t, binds=binds)
        if r is not None:
            return r

        if t == BaseTy(BaseTy.Scalar):
            if refine is None:
                return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
            else:
                return NamedCType(binds, refine)
        else:
            raise AssertionError(f"unrecognized type {repr(t)}")

def argument(a: Argument, *, refine: Optional[CType], tensor: bool) -> Optional[Binding]:
    nctype = argumenttype_type(a.type, binds=a.name, refine=refine, tensor=tensor)
    if nctype is None:
        return None
    return Binding(
        nctype=nctype,
        name=a.name,
        default=None,
        argument=a,
    )

def arguments(g: NativeFunctionsGroup, *, refine: Optional[CType], tensor: bool) -> List[Binding]:
    return list(mapMaybe(
        lambda a: argument(a, refine=refine, tensor=tensor),
        g.functional.func.arguments.flat_non_out
    ))
