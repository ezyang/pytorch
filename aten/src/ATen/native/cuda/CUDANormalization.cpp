#include <ATen/ATen.h>

namespace at { namespace native {

Tensor _batch_norm_cuda(
    const Tensor& input, const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    const Tensor& running_mean /* optional */, const Tensor& running_var /* optional */,
    bool training, double momentum, double eps, bool cudnn_enabled) {

  bool use_cudnn = false;
  use_cudnn = (input.type().is_cuda()
               && (input.type().scalarType() != at::kHalf
                 || weight.type().scalarType() == at::kFloat)
               && weight.defined() && bias.defined()
               && ((running_mean.defined() && running_var.defined())
                 || (!running_mean.defined() && !running_var.defined() && training))
               && input.size(0) <= 131070
               && detail::getCUDAHooks().compiledWithCuDNN()
               && cudnn_enabled && detail::getCUDAHooks().versionCuDNN() >= 5110L);

  if (use_cudnn && eps >= detail::getCUDAHooks().batchnormMinEpsilonCuDNN()) {
    return std::get<0>(at::cudnn_batch_norm(
                        input, weight, bias,
                        running_mean, running_var,
                        training, momentum, eps));
  }

  return at::thnn_batch_norm(
            input, weight, bias,
            running_mean, running_var, training, momentum, eps);
}

}} // at::native
