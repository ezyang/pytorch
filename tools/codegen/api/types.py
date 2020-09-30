from tools.codegen.model import *
from dataclasses import dataclass
from typing import Optional, Union, Sequence

# Functions only, no types
import tools.codegen.api.cpp as cpp

# Represents the implicit *this argument for method calls in C++ API
@dataclass(frozen=True)
class ThisArgument:
    argument: Argument

# Bundle of arguments that represent a TensorOptions in the C++ API.
@dataclass(frozen=True)
class TensorOptionsArguments:
    dtype: Argument
    layout: Argument
    device: Argument
    pin_memory: Argument

    def all(self) -> Sequence[Argument]:
        return [self.dtype, self.layout, self.device, self.pin_memory]

# Describe a argument (e.g., the x in "f(int x)") in the C++ API
@dataclass(frozen=True)
class CppArgument:
    # C++ type, e.g., int
    type: str
    # C++ name, e.g., x
    name: str
    # Only used by the header, but we work it out in all cases anyway
    default: Optional[str]
    # The JIT argument(s) this formal was derived from.  May
    # correspond to multiple arguments if this is TensorOptions!
    # May also correspond to the implicit *this argument!
    argument: Union[Argument, TensorOptionsArguments, ThisArgument]

    # Default string representation prints the most elaborated form
    # of the formal
    def __str__(self) -> str:
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"
        return f"{self.type} {self.name}{mb_default}"

    # However, you might also find the version with no default useful
    def str_no_default(self) -> str:
        return f"{self.type} {self.name}"

@dataclass(frozen=True)
class CppExpr:
    type: str
    expr: str

# A CppSignature is very similar to a FunctionSchema, but it is
# augmented with decisions about defaulting and overloads that are C++
# specific (for example, single functions in native functions
# may desugar into multiple C++ overloads).  There is a CppSignature
# per C++ overload, and this class contains enough information to
# distinguish between these overloads
@dataclass(frozen=True)
class CppSignature:
    # The schema this signature is derived from
    func: FunctionSchema
    # If this signature is a method, this is not None and contains
    # the corresponding ThisArgument for this signature
    method: Optional[ThisArgument]

    # Some cached stuff.  Kind of creeky
    cpp_arguments: Tuple[CppArgument, ...]
    cpp_return_type: str

    # Read-only on arguments important to enable covariance
    @staticmethod
    def from_arguments(
        func: FunctionSchema,
        arguments: Sequence[Argument, TensorOptionsArguments, ThisArgument],
        *,
        strip_defaults: bool
    ) -> 'CppSignature':

        def maybe_strip_default(a: CppArgument) -> CppArgument:
            if strip_defaults:
                return CppArgument(
                    type=a.type,
                    name=a.name,
                    default=None,
                    argument=a.argument,
                )
            else:
                return a

        return CppSignature(
            func=func,
            method=next((a for a in arguments if isinstance(a, ThisArgument)), None),
            cpp_arguments=tuple(
                maybe_strip_default(cpp.argument(a)) for a in arguments if not isinstance(a, ThisArgument)),
            cpp_return_type=cpp.returns_type(func.returns),
        )

@dataclass(frozen=True)
class CppSignatureGroup:
    func: FunctionSchema
    signature: CppSignature
    gathered_signature: Optional[CppSignature]

    def user_signature(self) -> CppSignature:
        if self.gathered_signature is not None:
            return self.gathered_signature
        else:
            return self.signature

    @staticmethod
    def from_schema(func: FunctionSchema, *, method: bool) -> 'CppSignatureGroup':
        r = cpp.gather_arguments(func, method=method)
        gathered_signature: Optional[CppSignature] = None
        r_arguments = r.arguments
        # BTW: this faffing about is a pretty good indication that
        # signature should be the optional one, and gathered signature
        # the non-optional one
        strip_defaults = False
        if r.gathered_arguments is not None:
            strip_defaults = True
            gathered_signature = CppSignature.from_arguments(func, r.gathered_arguments, method=method, strip_defaults = False)
        signature = CppSignature.from_arguments(func, r_arguments, method=method, strip_defaults=strip_defaults)
        return CppSignatureGroup(
            func=func,
            signature=signature,
            gathered_signature=gather_signature,
        )

@dataclass(frozen=True)
class DispatcherExpr:
    type: str
    expr: str

@dataclass(frozen=True)
class LegacyDispatcherExpr:
    type: str
    expr: str

@dataclass(frozen=True)
class DispatcherArgument:
    type: str
    name: str
    # dispatcher NEVER has defaults
    argument: Union[Argument, TensorOptionsArguments]
    # TensorOptionsArguments can occur when not using full c10 dispatch

    def __str__(self) -> str:
        return f"{self.type} {self.name}"

@dataclass(frozen=True)
class LegacyDispatcherArgument:
    type: str
    name: str
    # Legacy dispatcher arguments have defaults for some reasons (e.g.,
    # the function prototypes in CPUType.h are defaulted).  There isn't
    # really any good reason to do this, as these functions are only
    # ever called from a context where all defaulted arguments are
    # guaranteed to be given explicitly.
    # TODO: Remove this
    default: Optional[str]
    argument: Union[Argument, TensorOptionsArguments]

    # Convention here is swapped because arguably legacy
    # dispatcher shouldn't have defaults...
    def __str__(self) -> str:
        return f"{self.type} {self.name}"

    def str_with_default(self) -> str:
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"
        return f"{self.type} {self.name}{mb_default}"
