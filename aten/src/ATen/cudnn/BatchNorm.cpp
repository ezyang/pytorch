#include "BatchNorm.h"

#include "Descriptors.h"
#include "Types.h"
#include "Utils.h"

#include <ATen/Check.h>


namespace at { namespace cudnn {

namespace {

void batchnorm_shape_check(CheckedFrom c, TensorArg input, TensorArg output, TensorArg running_mean) {
  checkDimRange(c, input, 2, 6 /* exclusive */);
  checkDim(c, running_mean, 1);
  checkContiguous(c, running_mean);
  // TODO: Check the rest of the parameters
  // TODO: Check output
}

// TODO: Scale is float even when input is half

}  // namespace

Tensor cudnn_batch_norm_forward(
    const Tensor& input_t, const Tensor& weight_t,
    const Tensor& bias_t, const Tensor& running_mean_t, const Tensor& running_var_t,
    const Tensor& save_mean_t, const Tensor& save_var_t, bool training,
    double exponential_average_factor, double epsilon)
{
  TensorArg input{ input_t, "input", 1 },
            weight{ weight_t, "weight", 2 },
            bias{ bias_t, "bias", 3 },
            running_mean{ running_mean_t, "running_mean", 4 },
            running_var{ running_var_t, "running_var", 5 },
            save_mean{ save_mean_t, "save_mean", 6 },
            save_var{ save_var_t, "save_var", 7 };
  CheckedFrom c = "cudnn_batch_norm_forward";
  cudnnSetStreamToCurrent();
  checkSameType(c, {weight, bias, running_mean, running_var, save_mean, save_var});
  // Check that if input is half, weight is float (but otherwise, they
  // match)
  checkSameGPU(c, {input, weight, bias, running_mean, running_var, save_mean, save_var});
  cudnnBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
  } else {
    mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7003
    if(training)
      mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif
  }

  auto output_t = input->type().tensor();
  output_t.resize_(input->sizes());
  TensorArg output{ output_t, "output", 0 };

  // TODO: Don't forget to check shape compatibility
  batchnorm_shape_check(c, input, output, running_mean);

  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(*input);
  TensorDescriptor idesc{ *input, 4 };  // input descriptor
  TensorDescriptor odesc{ *output, 4 };  // output descriptor
  // TODO: For some reason, this was previously, inexplicably initialized from
  // running_mean.  Check if there was a reason.
  TensorDescriptor wdesc{ weight->expand({1, weight->size(0)}), std::max<int64_t>(4, input->dim()) };  // descriptor for weight, bias, running_mean, etc.

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  if (training) {
    checkContiguous(c, {input, bias, running_mean, running_var, save_mean, save_var});
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      handle, mode, &one, &zero,
      idesc.desc, idesc.ptr,
      odesc.desc, odesc.ptr,
      wdesc.desc, wdesc.ptr,
      bias->data_ptr(),
      exponential_average_factor,
      running_mean->data_ptr(),
      running_var->data_ptr(),
      epsilon,
      save_mean->data_ptr(),
      save_var->data_ptr()));
  } else {
    checkContiguous(c, {input, bias, running_mean, running_var});
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      handle, mode, &one, &zero,
      idesc.desc, idesc.ptr,
      odesc.desc, odesc.ptr,
      wdesc.desc, wdesc.ptr,
      bias->data_ptr(),
      running_mean->data_ptr(),
      running_var->data_ptr(),
      epsilon));
  }

  return output_t;
}

std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm_backward(
    const Tensor& input_t, const Tensor& grad_output_t, const Tensor& weight_t,
    const Tensor& running_mean_t, const Tensor& running_var_t,
    const Tensor& save_mean_t, const Tensor& save_var_t, bool training,
    double epsilon)
{
  TensorArg input{ input_t, "input", 1 },
            grad_output{ grad_output_t, "grad_output", 2 },
            weight{ weight_t, "weight", 3 },
            running_mean{ running_mean_t, "running_mean", 4 },
            running_var{ running_var_t, "running_var", 5 },
            save_mean{ save_mean_t, "save_mean", 6 },
            save_var{ save_var_t, "save_var", 7 };
  CheckedFrom c = "cudnn_batch_norm_backward";
  cudnnSetStreamToCurrent();
  checkSameType(c, {input, grad_output});
  checkSameType(c, {weight, running_mean, running_var, save_mean, save_var});
  checkSameGPU(c, {input, grad_output, weight, running_mean, running_var, save_mean, save_var});
  cudnnBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
  } else {
    mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7003
    if(training)
      mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif

  }

  auto grad_input_t = input->type().tensor();
  grad_input_t.resize_(input->sizes());
  auto grad_weight_t = weight->type().tensor();
  grad_weight_t.resize_(weight->sizes());
  auto grad_bias_t = weight->type().tensor();
  grad_bias_t.resize_(weight->sizes());

  checkContiguous(c, {input, grad_output, save_mean, save_var});

  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(*input);

  TensorDescriptor idesc{ *input, 4 };  // input descriptor
  TensorDescriptor odesc{ *grad_output, 4 };  // output descriptor
  TensorDescriptor gdesc{ grad_input_t, 4 };  // grad_input descriptor
  TensorDescriptor wdesc{ weight->expand({1, weight->size(0)}), std::max<int64_t>(4, input->dim()) };  // descriptor for weight, bias, running_mean, etc.

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  CUDNN_CHECK(cudnnBatchNormalizationBackward(
    handle, mode, &one, &zero, &one, &zero,
    idesc.desc, idesc.ptr,
    odesc.desc, odesc.ptr,
    gdesc.desc, gdesc.ptr,
    wdesc.desc, wdesc.ptr,
    grad_weight_t.data_ptr(),
    grad_bias_t.data_ptr(),
    epsilon,
    save_mean->data_ptr(),
    save_var->data_ptr()));

  return std::tuple<Tensor,Tensor,Tensor>{grad_input_t, grad_weight_t, grad_bias_t};
}

}}  // namespace cudnn
