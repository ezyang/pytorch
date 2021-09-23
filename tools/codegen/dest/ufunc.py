from dataclasses import dataclass
from typing import Union
from typing_extensions import Literal
from tools.codegen.api.translate import translate
from tools.codegen.utils import Target
from tools.codegen.model import (NativeFunctionsGroup)
import tools.codegen.api.ufunc as ufunc
import tools.codegen.api.structured as structured
from tools.codegen.api.types import StructuredImplSignature, scalar_t, opmath_t

# NB: not bothering to generate dispatch stub forward declaration in header,
# we can just paste it whereever necessary

# TODO: use BackendIndex
# dispatch_key: DispatchKey  # only CPU/CUDA right now


# Functors are templated because when USERS instantiate functors they are
# templated
@dataclass(frozen=True)
class UfunctorSignature:
    g: NativeFunctionsGroup
    scalar_tensor: Optional[int]
    name: str

    def arguments(self) -> UfunctorArguments:
        return ufunc.ufunctor_arguments(self.g, scalar_tensor=self.scalar_tensor)

    def fields(self) -> List[Binding]:
        raise NotImplementedError

    def returns_type(self) -> CType:
        # TODO: don't hardcode
        return scalar_t

    def decl_fields(self) -> str:
        raise NotImplementedError

    def decl_ctor(self) -> str:
        args_str = ', '.join(a.decl() for a in self.arguments().ctor)
        return f"{self.name}({args_str})";

    def decl_apply(self) -> str:
        args_str = ', '.join(a.decl() for a in self.arguments().apply)
        return f"{self.returns_type()} operator()({args_str})"


@dataclass(frozen=True)
class UfuncSignature:
    pass


# stuff for CPU

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




# steps:
#   1. take the functional signature
#   2. use api.ufunc to convert it to template signature.  this establishes
#      the type of the template function
#   3. use api.ufunc (II) to generate a split struct / operator() signature.
#      this establish context in which we call the template signature
#
# StructuredImplSignature context
#   ~> functor constructor sig
#
# Functor constructor context
#   ~> functor fields sig
#
# Functor apply context (functor fields + functor apply sig)
#   ~> template sig
#
# IDEA: don't use precanned templates at all, do it all by hand here

def eligible_for_binary_scalar_specialization(g: NativeFunctionsGroup) -> bool:
    assert False

def compute_ufunc_cuda_functors(g: NativeFunctionGroup) -> Tuple[Dict[ScalarType, Dict[UfuncKey, UfunctorSignature]], List[str]]:
    # First, build the functors.
    ufunctor_sigs: Dict[ScalarType, Dict[UfuncKey, UfunctorSignature]] = {}
    ufunctors: List[str] = []
    loops = g.functional.ufunc_inner_loop
    scalar_tensor_lookup = {
      UfuncKey.CUDAFunctorOnSelf: 1,
      UfuncKey.CUDAFunctorOnOther: 0,
      UfuncKey.CUDAFunctor: None
    }
    if eligible_for_binary_scalar_specialization(g):
        keys = [UfuncKey.CUDAFunctorOnSelf, UfuncKey.CUDAFunctorOnOther, UfuncKey.CUDAFunctor]
    else:
        keys = [UfuncKey.CUDAFunctor]
    for k in keys:
        if k in loops:
            supported_dtypes = loops[k].supported_dtypes
            name = loops[k].name
        else:
            supported_dtypes = loops[UfuncKey.Generic].supported_dtypes
            ufunc_name = loops[UfuncKey.Generic].name
            name = f"{k}_{ufunc_name}"

        ufunctor_sig = UfunctorSignature(g, scalar_tensor=scalar_tensor_lookup[k], name=name)
        for dtype in supported_dtypes:
            r = ufunctor_sigs.setdefault(dtype, {}).setdefault(k, ufunctor_sig)
            assert r is ufunctor_sig

        if k in loops:
            ufunc_sig = UfuncSignature(g, name=ufunc_name)

            apply_ctx = ufunctor_sig.fields() + ufunctor_sig.arguments().apply
            ufunc_exprs = translate(apply_ctx, ufunc_sig.arguments())
            ufunc_exprs_str = ', '.join(e.expr for e in ufunc_exprs)

            ufunctors.append(f"""
struct {ufunctor_sig.name} {{
  {ufunctor_sig.decl_fields()}
  {ufunctor_sig.decl_ctor()} {{}}
  {ufunctor_sig.decl_apply()} {{
    return {ufunc_sig.name}({ufunc_exprs_str});
  }}
}};
""")

    return ufunctor_sigs, ufunctors

@dataclass(frozen=True)
class BinaryScalarSpecializationConfig:
    scalar_idx: int
    ctor_tensor: str
    ufunc_key: UfuncKey

BinaryScalarSpecializationConfigs = [
    BinaryScalarSpecializationConfig(
        scalar_idx = 0,
        ctor_tensor = 'self',
        ufunc_key = UfuncKey.CUDAFunctorOnOther,
    ),
    BinaryScalarSpecializationConfig(
        scalar_idx = 1,
        ctor_tensor = 'other',
        ufunc_key = UfuncKey.CUDAFunctorOnSelf,
    ),
]

def compute_ufunc_cuda_per_dtype(g: NativeFunctionsGroup, dtype: ScalarType, inner_loops: Dict[UfuncKey, UfunctorSignature], parent_ctx: List[Union[Binding, Expr]]):
    body = "using opmath_t = at::opmath_type<scalar_t>;"
    body += "if (false) {{}}\n";
    for config in BinaryScalarSpecializationConfigs:
        if config.ufunc_key not in inner_loops:
            continue
        ufunctor_sig = inner_loop[config.ufunc_key]
        scalar_idx = ufunc.scalar_tensor + 1
        ctx = parent_ctx.clone()
        ctx.append(Expr(
            expr=f"__scalar_{ufunc.ctor_tensor}",
            type=NamedCType(ufunc.ctor_tensor, BaseCType(opmath_t)),
        ))
        ufunctor_ctor_exprs_str = ', '.join(a.expr for a in translate(ctx, ufunctor_sig.arguments().ctor))

        body += """\
else if (iter.is_cpu_scalar({scalar_idx})) {{
  auto __scalar_{ufunc.ctor_tensor} = iter.scalar_value<opmath_t>({scalar_idx});
  iter.remove_operand({scalar_idx});
  gpu_kernel(iter, {ufunctor_sig.name()}<scalar_t>({ufunctor_ctor_exprs_str}));
}}"""

    ufunctor_sig = inner_loop[UFuncKey.CUDAFunctor]
    ufunctor_ctor_exprs_str = ', '.join(a.expr for a in translate(ctx, ufunctor_sig.arguments().ctor))
    body += """
else {{
  gpu_kernel(iter, {ufunctor_sig.name()}<scalar_t>({ufunctor_ctor_exprs_str}));
}}
    """
    return InnerOuter(inner=body, outer="\n\n".join(functors))

def compute_ufunc_cuda(g: NativeFunctionsGroup) -> str:
    # First, build the functors, indexing them by dtype
    ufunctor_sigs, ufunctors = compute_ufunc_cuda_functors(g)

    # Next, build the conditionals
    sig = StructuredImplSignature(g, ufunc.kernel_name(g))
    dtype_cases = []
    for dtype, inner_ufunctor_sigs in ufunctor_sigs.items():
        dtype_cases.append(f"""
AT_PRIVATE_CASE_TYPE("{sig.name}", at::ScalarType::{dtype}, float,
  [&]() {{
    {compute_ufunc_cuda_dtype_body(g, dtype, inner_ufunctor_sigs, sig.arguments())}
  }}
)
""")

    dtype_cases_str = "\n".join(dtype_cases)

    return f"""
{sig.defn()} {{
  TensorIteratorBase& iter = *this;
  at::ScalarType st = iter.common_dtype();
  RECORD_KERNEL_FUNCTION_DTYPE("{sig.name}", st);
  switch (st) {{
    {dtype_cases_str}
    default:
      TORCH_CHECK(false, "{sig.name}", " not implemented for '", toString(st), "'");
  }}
}}
"""
