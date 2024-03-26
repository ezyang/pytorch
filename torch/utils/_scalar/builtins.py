import torch
import torch._prims_common

import torch._prims_common as utils

class TensorInt:
    tensor: torch.Tensor  # int64 dtype

    def __init__(self, t=None):
        if t is None:
            self.tensor = torch.scalar_tensor(0, dtype=torch.int64)
        elif isinstance(t, int):
            self.tensor = torch.scalar_tensor(t, dtype=torch.int64)
        elif isinstance(t, torch.Tensor):
            assert t.dtype == torch.int64
            self.tensor = t
        else:
            raise ValueError(f"unrecognized arg {t}")

    def _pytype(self):
        return int

    # Let's talk about Tensor-Scalar promotion rules.  Unfortunately,
    # we cannot read these off of PrimTorch decompositions, because our
    # prims support Scalar arguments for efficiency reasons.  Scalars in
    # Python are always stored in the highest precision possible: float64 or
    # int64.  However, this precision does not affect the final dtype of the
    # result.  Furthermore, we do the actual compute in the precision
    # specified by the Tensor (as opposed doing compute in high precision and
    # downcasting); you can observe this with:
    #
    #   >>> 16777217. - torch.tensor(16777216.)
    #   tensor(0.)
    #
    # Scalars do affect the type (bool, int, float).  So this means,
    # generically, the way we do a binary operation is to:
    #
    #  1. Compute the type using all arguments
    #  2. Compute the precision.  If no tensor has the same type, use the
    #  default precision.
    #
    # You might be tempted to convert the Python scalar into a Tensor scalar
    # and rely on the fact that we do not consider tensors with zero
    # dimensions when determining precision.  However, this leads to behavior
    # divergence when the *tensor* is also 0-dim:
    #
    #   >>> torch.tensor(2, dtype=torch.int32) + torch.tensor(3, dtype=torch.int64)
    #   tensor(5)
    #   >>> torch.tensor(2, dtype=torch.int32) + 3
    #   tensor(5, dtype=torch.int32)

    def __add__(self, other):
        binop = operator.add
        if isinstance(other, TensorInt):
            return TensorInt(binop(self.tensor, other.tensor))
        elif isinstance(other, torch.Tensor):
            other_type = utils.dtype_to_type(other.dtype)
            highest_type = get_higher_type(self._pytype(), other_type)
            if other.dtype is highest_type:
                result_dtype = other.dtype
            else:
                # Can't read off the type from Tensor, pick the default
                if highest_type is float:
                    result_dtype = torch.get_default_dtype()
                elif highest_type is complex:
                    result_dtype = utils.corresponding_complex_dtype(torch.get_default_dtype())
                elif highest_type is int:
                    result_dtype = torch.int64
                elif highest_type is bool:
                    result_dtype = torch.bool
        return NotImplemented

class TensorFloat:
    tensor: torch.Tensor  # float64 dtype

    def __init__(self, t=None):
        if t is None:
            self.tensor = torch.scalar_tensor(0.0, dtype=torch.float64)
        elif isinstance(t, float):
            self.tensor = torch.scalar_tensor(t, dtype=torch.float64)
        elif isinstance(t, torch.Tensor):
            assert t.dtype == torch.float64
            self.tensor = t
        else:
            raise ValueError(f"unrecognized arg {t}")

class TensorComplex:
    tensor: torch.Tensor  # complex128 dtype

    def __init__(self, t=None):
        if t is None:
            self.tensor = torch.scalar_tensor(False, dtype=torch.bool)
        elif isinstance(t, bool):
            self.tensor = torch.scalar_tensor(t, dtype=torch.bool)
        elif isinstance(t, torch.Tensor):
            assert t.dtype == torch.complex128
            self.tensor = t
        else:
            raise ValueError(f"unrecognized arg {t}")


class TensorBool(int):
    tensor: torch.Tensor  # bool dtype

    def __init__(self, t=None):
        if t is None:
            self.tensor = torch.scalar_tensor(False, dtype=torch.bool)
        elif isinstance(t, bool):
            self.tensor = torch.scalar_tensor(t, dtype=torch.bool)
        elif isinstance(t, torch.Tensor):
            assert t.dtype == torch.bool
            self.tensor = t
        else:
            raise ValueError(f"unrecognized arg {t}")
