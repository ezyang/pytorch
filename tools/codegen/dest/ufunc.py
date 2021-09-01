from dataclasses import dataclass
from typing import Union
from typing_extensions import Literal

from tools.codegen.utils import Target
from tools.codegen.model import (NativeFunctionsGroup)
import tools.codegen.api.ufunc as ufunc

# NB: not bothering to generate dispatch stub forward declaration in header,
# we can just paste it whereever necessary

# TODO: use BackendIndex
# dispatch_key: DispatchKey  # only CPU/CUDA right now




def stub_type(g: NativeFunctionsGroup) -> str:
    cpp_args = ufunc.arguments(g, refine=None, tensor=False)
    return f"void(*)(TensorIteratorBase&, {', '.join(a.type for a in cpp_args)})"

def stub_name(g: NativeFunctionsGroup) -> str:
    # TODO: verify no conflicts in BaseOperatorName for structured kernels
    return "{str(g.functional.func.name.name)}_stub"

def declare_dispatch(g: NativeFunctionsGroup) -> str:
    return f"DECLARE_DISPATCH({stub_type(g)}, {stub_name(g)});"

def define_dispatch(g: NativeFunctionsGroup) -> str:
    return f"""
{declare_dispatch(g)}
DEFINE_DISPATCH({stub_name(g)})
"""

# TODO: define impl

# stub typing: TensorIteratorBase&, omit all tensor arguments

@dataclass(frozen=True)
class UfuncOperator:
    # TODO: add context
    pass

"""
    def __call__(self, ufunc: Ufunc) -> str:
        return '''



'''
"""
