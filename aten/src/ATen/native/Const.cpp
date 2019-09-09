#include <ATen/ATen.h>

namespace at {
namespace native {

Tensor _const5(const Tensor& self, const Tensor& second, const Tensor& third, const Tensor& fourth, const Tensor& fifth) {
  return self;
}

}} // namespace at::native
