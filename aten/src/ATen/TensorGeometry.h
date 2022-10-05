#pragma once

#include <ATen/core/TensorBase.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/strides.h>

namespace at {

// Return if the tensor geometry represented by `sizes` and `strides` is
// contiguous Although we cache is_contiguous in tensor now, this is till useful
// because it allows checking if a particular geometry is contiguous without
// explicitly constructing a tensor, e.g., when you want to choose a kernel
// strategy based on whether a subgeometry is contiguous.
TORCH_API bool geometry_is_contiguous(IntArrayRef sizes, IntArrayRef strides);

struct TORCH_API TensorGeometry {
  TensorGeometry() : storage_offset_(0) {}

  explicit TensorGeometry(c10::SymIntArrayRef sizes)
      : sizes_(sizes),
        strides_(c10::contiguous_strides(sizes)),
        storage_offset_(0),
        numel_(c10::multiply_integers(sizes)),
        has_symbolic_sizes_strides_(sizes_.is_symbolic()) {
  }

  explicit TensorGeometry(const TensorBase& t)
      : sizes_(t.sym_sizes()),
        strides_(t.sym_strides()),
        storage_offset_(t.sym_storage_offset()),
        numel_(t.sym_numel()),
        has_symbolic_sizes_strides_(t.unsafeGetTensorImpl()->has_symbolic_sizes_strides())
        {}

  // true if the tensor is contiguous
  bool is_contiguous() const;

  int64_t dim() const {
    return sizes_.size();
  }

  int64_t size(int64_t dim) const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return sizes_.at(static_cast<size_t>(dim)).as_int_unchecked();
  }
  c10::IntArrayRef sizes() const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    return c10::asIntArrayRefUnchecked(sizes_);
  }
  int64_t stride(int64_t dim) const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return strides_.at(static_cast<size_t>(dim)).as_int_unchecked();
  }
  c10::IntArrayRef strides() const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    return c10::asIntArrayRefUnchecked(strides_);
  }
  int64_t storage_offset() const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    return storage_offset_.as_int_unchecked();
  }
  int64_t numel() const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    return numel_.as_int_unchecked();
  }

  c10::SymInt sym_size(int64_t dim) const {
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return sizes_.at(static_cast<size_t>(dim));
  }
  c10::SymIntArrayRef sym_sizes() const {
    return sizes_;
  }
  c10::SymInt sym_stride(int64_t dim) const {
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return strides_.at(static_cast<size_t>(dim));
  }
  c10::SymIntArrayRef sym_strides() const {
    return strides_;
  }
  c10::SymInt sym_storage_offset() const {
    return storage_offset_;
  }
  c10::SymInt sym_numel() const {
    return numel_;
  }

  TensorGeometry transpose(int64_t dim0, int64_t dim1) {
    TensorGeometry r = *this; // copy
    TORCH_CHECK(
        dim0 < dim(),
        "transpose: dim0=",
        dim0,
        " out of range (dim=",
        dim(),
        ")")
    TORCH_CHECK(
        dim1 < dim(),
        "transpose: dim1=",
        dim1,
        " out of range (dim=",
        dim(),
        ")")
    r.sizes_.transpose(dim0, dim1);
    r.strides_.transpose(dim0, dim1);
    return r;
  }

 private:
  c10::SymDimVectorWithIsSymbolic sizes_;
  c10::SymDimVectorWithIsSymbolic strides_;
  c10::SymInt storage_offset_;
  c10::SymInt numel_;
  bool has_symbolic_sizes_strides_;
};

} // namespace at
