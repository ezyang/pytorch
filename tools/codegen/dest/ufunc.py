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


# steps:
#   1. take the functional signature
#   2. use api.ufunc to convert it to template signature.  this establishes
#      the type of the template function
#   3. use api.ufunc (II) to generate a split struct / operator() signature.
#      this establish context in which we call the template signature
#
#
#
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

@dataclass(frozen=True)
class InnerOuter:
    inner: str
    outer: str

def compute_ufunc_cuda_per_dtype(g: NativeFunctionsGroup, dtype: ScalarType, parent_ctx: List[Union[Binding, Expr]]) -> InnerOuter:
    functors = []
    body = "using opmath_t = at::opmath_type<scalar_t>;"
    body += "if (false) {{}}\n";
    if eligible_for_binary_scalar_specialization(g):
        # TODO: in principle, can support scalar specialization for arbitrary
        # n-ary operators, but if you actually want to do it also have to update
        # TensorIterator to accept this
        for scalar_tensor, tensor_name in enumerate(['self', 'other']):
            scalar_idx = scalar_tensor + 1
            ctx = parent_ctx.clone()
            ctx.append(Expr(
                expr=f"__scalar_{tensor_name}",
                type=NamedCType(tensor_name, BaseCType(opmath_t)),
            ))
            # find the correct inner loop to use here
            inner_loop: Optional[UfuncInnerLoop] = None
            for loop in g.functional.ufunc_inner_loop:
                if dtype not in loop.supported_dtypes:
                    continue
                # NB: intentionally flipped: if we make other a scalar, then the
                # functor takes self as argument
                if (tensor_name == 'other' and loop.ufunc_key is UfuncKey.CUDAFunctorOnSelf) or \
                        loop.ufunc_key is UfuncKey.Generic:
                    inner_loop = loop
                    break
            assert inner_loop is not None, \
                f"was expecting to specialize {tensor_name} to scalar, but no eligible ufunc was found"

            functor_sig = UfunctorSignature(g, scalar_tensor=scalar_tensor, f"TODOTODO")
            if inner_loop.ufunc_key is UfuncKey.Generic:
                pass

            body += """\
else if (iter.is_cpu_scalar({scalar_idx})) {{
  auto __scalar_{tensor_name} = iter.scalar_value<opmath_t>({scalar_idx});
  iter.remove_operand({scalar_idx});
}}"""
    body += ""
    return InnerOuter(inner=body, outer="\n\n".join(functors))

def compute_ufunc_cuda(g: NativeFunctionsGroup) -> str:
    # struct Functor {
    #   Functor(... UFunctor ...) {
    #   }
    # }

    sig = StructuredImplSignature(g, ufunc.kernel_name(g))

    dtype_cases = []
    for dtype in ScalarType:
        # check if we have ANY implementation of the dtype.  We'll get
        # more detailed shortly
        # WARNING: quadratic police.  could probably preindex but whatever
        for loop in g.functional.ufunc_inner_loop:
            if dtype in loop.supported_dtypes:
                break
        else:
            continue
        dtype_cases.append(f"""
AT_PRIVATE_CASE_TYPE("{sig.name}", at::ScalarType::{dtype}, float,
  [&]() {{
    {compute_ufunc_cuda_dtype_body(g, dtype, sig.arguments())}
  }}
)
""")


'''
    dtype_cases = []
    functors = []

    for dtype, kernel in metadata.dtype_dispatch.items():
        assert isinstance(kernel, UfuncCUDAKernel)

        loop_body = ""
        loop_body += "if (false) {{}}\n"
        # TODO: we need to know statically the dtype information about
        # the output to know what target dtype we should have.  for now,
        # assume output dtype is same as input
        if kernel.scalar_tensor_fn is not None:
            fn = kernel.scalar_tensor_fn
            if isinstance(fn, UfuncGenericFn):
                # If a GenericFn, I have a template that when instantiated
                # with a scalar_t template argument, takes two scalar_t
                # and then returns a scalar_t
                #   NO!  Partial application!  In fact I have something
                #   that obeys api.ufunc.  Need to create the class
                #   for partial application
                # TODO: verify that this is a supported function
                raise NotImplemented
            elif isinstance(fn, UfuncSpecializedFn):
                ufunc_sig = UfuncSignature(g=g, name=fn.fn)
                ufunctor_sig = UfunctorSignature(g=g, scalar_tensor=0, name="unary TODO TODO")
                apply_ctx = ufunctor_sig.fields() + ufunctor_sig.arguments().apply
                ufunc_exprs = translate(apply_ctx, ufunc_sig.arguments())
                ufunc_exprs_str = ', '.join(e.expr for e in ufunc_exprs)
                functors.append(f"""
struct {ufunctor_sig.name} {{
  {ufunctor_sig.decl_fields()}
  {ufunctor_sig.decl_ctor()} {{}}
  {ufunctor_sig.decl_apply()} {{
    return {ufunc_sig.name}({ufunc_exprs_str});
  }}
}};
""")
            else:
                assert_never(fn)
            loop_body += f"""else if (iter.is_cpu_scalar(1)) {{
  auto functor = (iter.scalar_value<opmath_t>(1));
  iter.remove_operand(1);
  {inner}
}}"""
        if kernel.tensor_scalar_fn is not None:
            inner = ""
            loop_body += f"""else if (iter.is_cpu_scalar(2)) {{
  iter.remove_operand(2);
  {inner}
}}"""
        loop_body += f"""else {{
    gpu_kernel(make_binary_functor<scalar_t>({kernel.all_tensor_fn}<at::opmath_type<scalar_t>>));
}}
"""


    dtype_cases_str = "\n".join(dtype_cases)

    # TODO: need to define the one-off structs

    # TODO: assert that ufunc inherits from TensorIteratorBase
    return f"""

{sig.defn()} {{
  at::ScalarType st = this->common_dtype();
  RECORD_KERNEL_FUNCTION_DTYPE("{sig.name}", st);
  switch (st) {{
    {dtype_cases_str}
    default:
      TORCH_CHECK(false, "{sig.name}", " not implemented for '", toString(st), "'");
  }}
}}
"""
'''
