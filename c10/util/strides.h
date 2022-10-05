#pragma once
#include <c10/util/ArrayRef.h>
#include <c10/util/DimVector.h>

namespace c10 {

// Computes the contiguous strides of a tensor, given its sizes.
template <typename T>
inline SmallVector<T, kDimVectorStaticSize> _contiguous_strides(const ArrayRef<T> sizes) {
  int64_t dims = sizes.size();

  // With this intialisation we get the case dim == 0 or 1 right
  SmallVector<T, kDimVectorStaticSize> strides(dims, 1);

  for (auto i = dims - 2; i >= 0; --i) {
    // Strides can't be 0 even if sizes are 0.
    strides[i] = strides[i + 1] * std::max(sizes[i + 1], T{1});
  }

  return strides;
}

DimVector contiguous_strides(IntArrayRef sizes) { return _contiguous_strides(sizes); }
SymDimVector contiguous_strides(SymIntArrayRef sizes) { return _contiguous_strides(sizes); }

} // namespace c10
