#pragma once

#include <ATen/ATen.h>
#include "THC/THC.h"
#include "cudnn-wrapper.h"
#include "Handles.h"

namespace at { namespace cudnn {

inline void cudnnSetStreamToCurrent() {
  // TODO: Should we call lazyInitCUDA() or access thc_state directly?
  // TODO: Should getCurrentStream be a method on Context?
  cudnnSetStream(getCudnnHandle(), THCState_getCurrentStream(globalContext().lazyInitCUDA()));
}

// TODO: Move this out to ATen proper?
inline Tensor contiguousIfZeroInStrides(const Tensor& t) {
  for (auto s : t.strides()) {
    if (s == 0) return t.contiguous();
  }
  return t;
}

}}
