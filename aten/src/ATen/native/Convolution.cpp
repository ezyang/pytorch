#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/ExpandUtils.h"
#include <functional>
#include <numeric>

#if AT_CUDA_ENABLED()
#include "THC/THC.h"
#ifdef WITH_CUDNN
#include <ATen/cudnn/cudnn-wrapper.h>
#endif
#endif

namespace at {
namespace native {

struct ConvParams {
  // Unconditional vector copy and allocation! Drat!
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool transposed;
  std::vector<int64_t> output_padding;
  int groups;
  bool benchmark;
  bool deterministic;
  bool cudnn_enabled;

  bool is_strided() const;
  bool is_dilated() const;
  bool is_padded() const;
  bool is_output_padding_neg() const;
  bool is_output_padding_big() const;
  bool is_padding_neg() const;
  void view1d_as_2d();
  bool use_cudnn(const at::Tensor& input) const;
  bool use_nnpack(const at::Tensor& input) const;
  bool is_depthwise(const at::Tensor& input, const at::Tensor& weight) const;
};

auto ConvParams::is_strided() const -> bool {
  bool is_strided = false;
  for (int s : stride) {
    is_strided |= (s != 1);
  }
  return is_strided;
}

auto ConvParams::is_dilated() const -> bool {
  bool is_dilated = false;
  for (int d : dilation) {
    is_dilated |= (d != 1);
  }
  return is_dilated;
}

auto ConvParams::is_padded() const -> bool {
  bool is_padded = false;
  for (int p : padding) {
    is_padded |= (p != 0);
  }
  return is_padded;
}

auto ConvParams::is_output_padding_neg() const -> bool {
  bool is_non_neg = false;
  for (int p : output_padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

auto ConvParams::is_output_padding_big() const -> bool {
  bool is_big = false;
  for (size_t i = 0; i < output_padding.size(); i++) {
    is_big |= (output_padding[i] >= stride[i] || output_padding[i] >= dilation[i]);
  }
  return is_big;
}

auto ConvParams::is_padding_neg() const -> bool {
  bool is_non_neg = false;
  for (int p : padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}


auto ConvParams::view1d_as_2d() -> void {
  if (stride.size() == 1) {
    stride.insert(stride.begin(), 1);
    padding.insert(padding.begin(), 0);
    dilation.insert(dilation.begin(), 1);
    output_padding.insert(output_padding.begin(), 0);
  }
}

auto ConvParams::use_cudnn(const at::Tensor& input) const -> bool {
#if AT_CUDNN_ENABLED()
  if (!input.type().is_cuda() || !cudnn_enabled) {
    return false;
  }
  if (deterministic && is_dilated()) {
    // cudnn doesn't support deterministic dilated convolution fully yet
    return false;
  }
  if (is_dilated()) {
    // TODO: ATen-ify this
    cudaDeviceProp* prop = THCState_getCurrentDeviceProperties(globalContext().thc_state);
    // NOTE: extra parenthesis around numbers disable clang warnings about dead code
    return ((CUDNN_VERSION >= (6021)) || (CUDNN_VERSION >= (6000) && prop->major >= 5)) && !is_output_padding_big();
  }
  return !is_output_padding_big();
#endif
  return false;
}

// TODO: fix this
auto ConvParams::use_nnpack(const at::Tensor& input) const -> bool {
#ifdef WITH_NNPACK
  return input.type().ID() == at::TypeID::CPUFloat && // only on CPU Float Tensors
         !is_strided() && // doesn't support strides
         !is_dilated() && // or dilation
         !transposed &&   // or transposed tensors
         input.ndimension() == 4 && // must be in NCHW format
         input.size(0) >= 16; // ensure large enough batch size to ensure perf, tuneable
#endif
  return false;
}

// We currently only have depthwise support for the case where groups ==
// nInputPlane and nInputPlane == nOutputPlane (the latter due to the lack of
// a depthwise multiplier)
auto ConvParams::is_depthwise(
        const at::Tensor& input, const at::Tensor& weight) const -> bool {
  return input.type().is_cuda() &&
         !transposed &&
         input.ndimension() == 4 &&
         input.size(1) == groups &&
         groups > 1 && // no point if there is only a single group
         weight.size(0) % input.size(1) == 0; // output channels must be a multiple of input channels
}


static auto view4d(const at::Tensor& tensor) -> at::Tensor {
  if (tensor.ndimension() != 3) throw std::runtime_error("expected 3D tensor");
  return tensor.unsqueeze(2);
}

static auto view3d(const at::Tensor& tensor) -> at::Tensor {
  if (tensor.ndimension() != 4) throw std::runtime_error("expected 4D tensor");
  return tensor.squeeze(2);
}

static at::Tensor subtensor(const at::Tensor& tensor, int dim, int groups, int g) {
  if (!tensor.defined()) {
    return at::Tensor();
  }
  int64_t n = tensor.sizes()[dim] / groups;
  return tensor.narrow(dim, n * g, n).contiguous();
}

// TODO: Use the built-in ATen cat
static at::Tensor cat(TensorList tensors, int dim) {
  int num_inputs = tensors.size();
  if (num_inputs == 0) {
    return at::Tensor();
  }

  auto output = tensors[0].type().tensor();
  at::cat_out(output, tensors, dim);
  return output;
}

static at::Tensor compute_output(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& columns, const at::Tensor& ones,
    const ConvParams& params) {

  auto dim = input.ndimension();
  auto dilated = params.is_dilated();
  auto kernel_size = weight.sizes().slice(2);

  if (params.transposed) {
    if (dim == 4) {
      return at::conv_transpose2d_forward(
          input, weight, kernel_size, bias,
          params.stride, params.padding, params.output_padding, params.dilation,
          columns, ones);
    } else if (dim == 5) {
      return at::conv_transpose3d_forward(
        input, weight, bias,
        params.stride, params.padding, params.output_padding, params.dilation,
        columns, ones);
      }
  } else {  /* Not transposed */
    if (dim == 4) {
      if (dilated) {
        return at::conv_dilated2d_forward(
            input, weight, kernel_size, bias,
            params.stride, params.padding, params.dilation,
            columns, ones);
      } else {  /* dim == 4, non-dilated */
        if (params.use_nnpack(input)) {
#ifdef WITH_NNPACK
          // THNN functions handle resizing the output Tensor themselves,
          // but NNPACK expects the Tensors to be in the appropriate shape
          // already, so we resize here
          auto output = input.type().tensor(params.output_size(input, weight));
          nnpack::SpatialConvolution_updateOutput(
              input, output, weight, bias,
              kernel_size[1], kernel_size[0],
              params.padding[1], params.padding[0]);
          return output;
#endif
        } else {
          /* CPU implementation has specialized MM kernels
             for non-dilated case here */
          return at::conv2d_forward(
              input, weight, kernel_size, bias,
              params.stride, params.padding,
              columns, ones);
        }
      }
    } else if (dim == 5 && (input.type().is_cuda() || dilated)) {
      return at::conv_dilated3d_forward(
          input, weight, kernel_size, bias,
          params.stride, params.padding, params.dilation,
          columns, ones);
    } else if (dim == 5) { /* dim == 5, CPU, non-dilated */
      /* CPU implementation has specialized MM kernels
         for non-dilated case here */
      return at::conv3d_forward(
          input, weight, kernel_size, bias,
          params.stride, params.padding,
          columns);
    }
  }

  throw std::runtime_error("unsupported ConvNd parameters");
}

// TODO: Rename the THNN convolution away from conv2d_* and
// take that namespace
std::tuple<at::Tensor, at::Tensor, at::Tensor> generic_convolution(
    // Alas, refcount bump needed here :(
    const Tensor& input_r, const Tensor& weight_r, const Tensor& bias,
    IntList stride_, IntList padding_, IntList dilation_,
    bool transposed_, IntList output_padding_,
    int64_t groups_, bool benchmark_, bool deterministic_, bool cudnn_enabled_) {

  ConvParams params;
  params.stride = stride_;
  params.padding = padding_;
  params.dilation = dilation_;
  params.transposed = transposed_;
  params.output_padding = output_padding_;
  params.groups = groups_;
  params.benchmark = benchmark_;
  params.deterministic = deterministic_;
  params.cudnn_enabled = cudnn_enabled_;

  // TODO: better error message here
  if (params.is_padding_neg()) throw std::runtime_error("negative padding is not supported");
  if (params.is_output_padding_neg()) throw std::runtime_error("negative output_padding is not supported");

  //check_input_shape_forward(input, weight, bias, groups, transposed);

  // input.contiguous()

  int k = input_r.ndimension();
  if (k == 3) {
    params.view1d_as_2d();
  }
  const auto& input = k == 3 ? view4d(input_r) : input_r;
  const auto& weight = k == 3 ? view4d(weight_r) : weight_r;

  // TODO: push these allocations into THNN.  They are vestiges from
  // when Lua didn't have a caching allocator.  We do now.
  Tensor columns = input.type().tensor();
  Tensor ones = input.type().tensor();

  Tensor output;
  if (params.is_depthwise(input, weight)) {
      auto kernel_size = weight.sizes().slice(2);
      output = at::conv_depthwise2d_forward(input, weight, kernel_size, bias,
                params.stride, params.padding, params.dilation);
  } else if (params.use_cudnn(input)) {
    if (params.transposed) {
      output = at::cudnn_convolution_transpose_full_forward(
          input, weight, bias,
          params.padding, params.output_padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
    } else {
      output = at::cudnn_convolution_full_forward(
          input, weight, bias,
          params.padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
    }
  } else {
    if (params.groups == 1) {
      output = compute_output(
          input, weight, bias,
          columns, ones, params);
    } else {
      std::vector<Tensor> outputs(params.groups);
      for (int g = 0; g < params.groups; ++g) {
        auto input_g = subtensor(input, 1, params.groups, g);
        auto weight_g = subtensor(weight, 0, params.groups, g);
        auto bias_g = subtensor(bias, 0, params.groups, g);
        outputs[g] = compute_output(
            input_g, weight_g, bias_g,
            columns, ones, params);
      }
      output = cat(outputs, 1);
    }
  }

  if (k == 3) {
    output = view3d(output);
  }

  return std::tuple<Tensor,Tensor,Tensor>{output, columns, ones};
}

// For Convolution strategies that don't implicitly handle grad_bias, we add a helper
// function here to perform it using simple Tensor operators
static at::Tensor compute_grad_bias(const at::Tensor& grad_output) {
  // grad_output is in N, C, H, W, we re-shape and reduce over spatial dims and batches
  return grad_output.contiguous().view({grad_output.size(0), grad_output.size(1), -1}).sum(0).sum(1);
}

static std::tuple<Tensor, Tensor, Tensor> compute_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    const at::Tensor& columns, const at::Tensor& ones,
    const ConvParams& params,
    std::array<bool, 3> output_mask) {

  auto kernel_size = weight.sizes().slice(2);
  auto stride = params.stride;
  auto padding = params.padding;
  auto dilation = params.dilation;
  auto output_padding = params.output_padding;

  auto dim = input.ndimension();
  auto dilated = params.is_dilated();

 if (params.transposed) {
    if (dim == 4) {
      return at::conv_transpose2d_backward(
          grad_output, input, weight, kernel_size,
          stride, padding, output_padding, dilation,
          columns, ones, output_mask);
    } else if (dim == 5) {
      return at::conv_transpose3d_backward(
          grad_output, input, weight,
          stride, padding, output_padding, dilation,
          columns, ones, output_mask);
    }
  } else {  /* Not transposed */
    if (dim == 4) {
      if (dilated) {
        return at::conv_dilated2d_backward(
            grad_output, input, weight, kernel_size,
            stride, padding, dilation,
            columns, ones, output_mask);
      } else {
        if (params.use_nnpack(input)) {
#ifdef WITH_NNPACK
          Tensor grad_input;
          Tensor grad_weight;
          Tensor grad_bias;

          if (output_mask[0]) {
            grad_input = input.type().tensor(input.sizes());
            nnpack::SpatialConvolution_updateGradInput(
                input, grad_output, grad_input, weight,
                kernel_size[1], kernel_size[0],
                params.padding[1], params.padding[0]);
          }

          // NNPACK does not have a bias gradient calculation, so we split
          // into two calls here if necessary
          if (output_mask[1]) {
            grad_weight = weight.type().tensor(weight.sizes());
            grad_weight.zero_();
            nnpack::SpatialConvolution_accGradWeight(
                input, grad_output, grad_weight,
                kernel_size[1], kernel_size[0],
                params.padding[1], params.padding[0]);
          }

          if (output_mask[2]) {
            grad_bias = compute_grad_bias(grad_output);
          }

          return std::make_tuple(grad_input, grad_weight, grad_bias);
#endif
        } else {
          /* CPU implementation has specialized MM kernels
             for non-dilated case here */
          return at::conv2d_backward(
              grad_output, input, weight, kernel_size,
              stride, padding,
              columns, ones, output_mask);
        }
      }
    } else if (dim == 5 && (input.type().is_cuda() || dilated)) {
        return at::conv_dilated3d_backward(
            grad_output, input, weight, kernel_size,
            stride, padding, dilation,
            columns, ones, output_mask);
    } else if (dim == 5) { /* dim == 5, CPU, non-dilated */
        /* CPU implementation has specialized MM kernels
           for non-dilated case here */
        return at::conv3d_backward(
            grad_output, input, weight, kernel_size,
            stride, padding,
            columns, ones, output_mask);
    }
  }

  throw std::runtime_error("unsupported ConvNdBackward parameters");
}


// Underscore to prevent THNN from triggering (remove me when
// THNN special case dies.
std::tuple<Tensor, Tensor, Tensor> _generic_convolution_backward(
    const at::Tensor& input_r, const at::Tensor& grad_output_r, const at::Tensor& weight_r,
    IntList stride_, IntList padding_, IntList dilation_,
    bool transposed_, IntList output_padding_,
    int64_t groups_, bool benchmark_, bool deterministic_, bool cudnn_enabled_,
    const at::Tensor& columns, const at::Tensor& ones,
    std::array<bool, 3> output_mask) {

  ConvParams params;
  params.stride = stride_;
  params.padding = padding_;
  params.dilation = dilation_;
  params.transposed = transposed_;
  params.output_padding = output_padding_;
  params.groups = groups_;
  params.benchmark = benchmark_;
  params.deterministic = deterministic_;
  params.cudnn_enabled = cudnn_enabled_;

  // input.contiguous()

  Tensor input = input_r, grad_output = grad_output_r, weight = weight_r;
  int k = input.ndimension();
  if (k == 3) {
    input = view4d(input);
    weight = view4d(weight);
    grad_output = view4d(grad_output);
  }

  bool use_depthwise = params.is_depthwise(input, weight);
  bool use_cudnn = params.use_cudnn(input);

  Tensor grad_input;
  at::Tensor grad_weight;
  at::Tensor grad_bias;

  if (use_depthwise) {
    if (output_mask[0] || output_mask[1]) {
      auto kernel_size = weight.sizes().slice(2);

      std::tie(grad_input, grad_weight) = at::conv_depthwise2d_backward(
          grad_output, input, weight, kernel_size, params.stride, params.padding, params.dilation,
          {{output_mask[0], output_mask[1]}});
      }

      // THCUNN implementation does not handle bias, so we do it ourselves
      if (output_mask[2]) {
        grad_bias = compute_grad_bias(grad_output);
      }
    } else if (use_cudnn) {
#if AT_CUDNN_ENABLED()
    if (output_mask[0]) {
      if (params.transposed) {
        grad_input = at::cudnn_convolution_transpose_backward(
            grad_output, weight,
            params.padding, params.stride, params.dilation, params.groups,
            params.benchmark, params.deterministic);
      } else {
        grad_input = at::cudnn_convolution_backward(
            input.sizes(), grad_output, weight,
            params.padding, params.stride, params.dilation, params.groups,
            params.benchmark, params.deterministic);
      }
    }
    if (output_mask[1] || output_mask[2]) {
      if (params.transposed) {
        grad_weight = at::cudnn_convolution_transpose_backward_weight(
            weight.sizes(), grad_output, input,
            params.padding, params.stride, params.dilation, params.groups,
            params.benchmark, params.deterministic);
      } else {
        grad_weight = at::cudnn_convolution_backward_weight(
            weight.sizes(), grad_output, input,
            params.padding, params.stride, params.dilation, params.groups,
            params.benchmark, params.deterministic);
      }

      if (output_mask[2]) {
        grad_bias = at::cudnn_convolution_backward_bias(grad_output);
      }
    }
#endif
  } else if (params.groups == 1) {
    std::tie(grad_input, grad_weight, grad_bias) = compute_backward(
        input, grad_output, weight, columns, ones,
        params, output_mask);
  } else {
    std::vector<Tensor> grad_inputs(params.groups);
    std::vector<Tensor>  grad_weights(params.groups);
    std::vector<Tensor>  grad_biases(params.groups);
    // TODO: maybe need to initialize columns/ones
    for (int g = 0; g < params.groups; ++g) {
      auto input_g = subtensor(input, 1, params.groups, g);
      auto grad_output_g = subtensor(grad_output, 1, params.groups, g);
      auto weight_g = subtensor(weight, 0, params.groups, g);
      std::tie(grad_inputs[g], grad_weights[g], grad_biases[g]) = compute_backward(
          input_g, grad_output_g, weight_g, columns, ones,
          params, output_mask);
    }
    if (output_mask[0]) {
      grad_input = cat(grad_inputs, 1);
    }
    if (output_mask[1]) {
      grad_weight = cat(grad_weights, 0);
    }
    if (output_mask[2]) {
      grad_bias = cat(grad_biases, 0);
    }
  }

  if (k == 3) {
    if (grad_input.defined()) {
      grad_input = view3d(grad_input);
    }
    if (grad_weight.defined()) {
      grad_weight = view3d(grad_weight);
    }
  }

  return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};


}


}
}
