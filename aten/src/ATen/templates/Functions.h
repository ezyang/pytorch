#pragma once

// ${generated_comment}

#include "ATen/core/Scalar.h"
#include "ATen/Type.h"
#include "ATen/TypeInternalInterface.h"
#include "ATen/Tensor.h"
#include "ATen/core/Storage.h"
#include "ATen/core/Generator.h"
#include "ATen/core/Deprecated.h"
#include "ATen/NativeFunctions.h"
#include "ATen/DeviceGuard.h"
#include "ATen/core/TensorOptions.h"
#include "ATen/core/Reduction.h"

namespace at {

using native::from_blob;
using native::tensor;

${function_declarations}

static inline TypeInternalInterface & infer_type(const Tensor & t) {
  AT_CHECK(t.defined(), "undefined Tensor");
  return dynamic_cast<TypeInternalInterface&>(t.type());
}
static inline TypeInternalInterface & infer_type(const TensorList & tl) {
  AT_CHECK(tl.size() > 0, "expected a non-empty list of Tensors");
  return dynamic_cast<TypeInternalInterface&>(tl[0].type());
}
static inline TypeInternalInterface & non_specific_type() {
  return dynamic_cast<TypeInternalInterface&>(at::getNonVariableType(at::Backend::Undefined, at::ScalarType::Float));
}
// function definitions are all static inline because
// they are one-line statically dispatched functions that
// invoke the actual dynamic dispatch on the correct argument
${function_definitions}

}
