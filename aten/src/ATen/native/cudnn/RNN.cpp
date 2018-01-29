#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/MatrixRef.h>

#if !AT_CUDNN_ENABLED()

namespace at { namespace native {

// See Note [ATen preprocessor philosophy]

Tensor _cudnn_rnn_flatten_weight(
    TensorList weight_arr, int64_t weight_stride0,
    int64_t input_size,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first,
    bool fn_bidirectional
    ) {
  throw std::runtime_error("_cudnn_rnn_flatten_weight: ATen not compiled with cuDNN support");
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _cudnn_rnn(
    const Tensor& input_r,
    TensorList weight, int64_t weight_stride0,
    const Tensor& weight_buf_r, const Tensor& hx, const Tensor& cx,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntList fn_batch_sizes,
    const Tensor& fn_dropout_state
    ) {
  throw std::runtime_error("_cudnn_rnn: ATen not compiled with cuDNN support");
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> _cudnn_rnn_backward(
    const Tensor& input, TensorList weight, int64_t weight_stride0, const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
    const Tensor& output, const Tensor& grad_output_r, const Tensor& grad_hy_r,
    const Tensor& grad_cy_r,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool batch_first, double dropout,
    bool train, bool bidirectional, IntList batch_sizes,
    const Tensor& dropout_state, const Tensor& reserve,
    std::array<bool, 4> output_mask
    ) {
  throw std::runtime_error("_cudnn_rnn_backward: ATen not compiled with cuDNN support");
}

}} // namespace at::native

#else // AT_CUDNN_ENABLED()

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>

namespace at { namespace native {

namespace {
  // DropoutDescriptor

  struct DropoutDescriptorParams {
    bool train;
    double dropout;
    Tensor dropout_state;
    DropoutDescriptorParams() {}
    void set(bool train_, double dropout_, Tensor dropout_state_) {
      train = train_;
      dropout = dropout_;
      dropout_state = dropout_state_;
    }
    DropoutDescriptor descriptor(cudnnHandle_t handle) const {
      // NB: dropout_seed passed dummy 0, because it isn't actually used
      // when dropout_state is defined.
      auto dropout_p = train ? dropout : 0;
      DropoutDescriptor dropout_desc;
      dropout_desc.set(handle, dropout_p, dropout_state, 0);
      return dropout_desc;
    }
  };

  // RNNDescriptor

  struct RNNDescriptorParams {
    int64_t hidden_size;
    int64_t num_layers;
    cudnnDirectionMode_t bidirectional;
    cudnnRNNMode_t mode;
    cudnnDataType_t datatype;

    cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;

    int64_t num_directions() const {
      return bidirectional ? 2 : 1;
    }

    void set_mode(int64_t fn_mode) {
      switch (fn_mode) {
        case CUDNN_RNN_RELU:
          mode = CUDNN_RNN_RELU;
          break;
        case CUDNN_RNN_TANH:
          mode = CUDNN_RNN_TANH;
          break;
        case CUDNN_LSTM:
          mode = CUDNN_LSTM;
          break;
        case CUDNN_GRU:
          mode = CUDNN_GRU;
          break;
        default:
          throw std::runtime_error("unrecognized mode"); // TODO
      }
    }

    void set_bidirectional(bool fn_bidirectional) {
      bidirectional = fn_bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    }

    RNNDescriptor descriptor(cudnnHandle_t handle, DropoutDescriptor&& dropout_desc) const {
      RNNDescriptor rnn_desc;
      rnn_desc.set(handle, hidden_size, num_layers, std::move(dropout_desc), input_mode, bidirectional, mode, datatype);
      return rnn_desc;
    }

    RNNDescriptor descriptor(cudnnHandle_t handle) const {
      DropoutDescriptor dropout_desc;
      dropout_desc.set(handle, 0, {}, 0);
      return descriptor(handle, std::move(dropout_desc));
    }
  };

  // TensorDescriptor list

  std::vector<TensorDescriptor> rnn_descriptor_sequence(const Tensor& tensor, IntList batch_sizes) {
    std::vector<TensorDescriptor> descriptors(batch_sizes.size());
    size_t i = 0;
    for (auto batch_size : batch_sizes) {
      // NB: The narrow is solely to adjust the batch size; to do it
      // accurately we would have to adjust the start index as well,
      // but the pointer location isn't actually used so we can skip it.
      // NB: cuDNN RNN API has an undocumented requirement that all
      // tensors have dimension 5.
      descriptors[i].set(tensor.narrow(0, 0, batch_size), 5);
      i++;
    }
    return descriptors;
  }

  std::vector<TensorDescriptor> rnn_descriptor(const Tensor& tensor, int64_t N) {
    std::vector<TensorDescriptor> descriptors(N);
    for (int64_t i = 0; i < N; i++) {
      descriptors[i].set(tensor, 5);
    }
    return descriptors;
  }

  struct TensorDescriptorListParams {
    IntList batch_sizes;
    int64_t seq_length;
    int64_t mini_batch;
    int64_t inner_size;  // previously known as "input_size"
    int64_t outer_size;  // only valid when !is_input_packed

    bool is_input_packed() const {
      return batch_sizes.size() != 0;
    }

    void set(IntList input_size, IntList batch_sizes_, bool batch_first) {
      batch_sizes = batch_sizes_;
      if (is_input_packed()) {
        seq_length = batch_sizes.size();
        mini_batch = batch_sizes[0];
        // NB: When input is packed, the mini_batch size is NOT the size
        // of the outer dimension
        outer_size = input_size[0];
        inner_size = input_size[1];
      } else {
        if (batch_first) {
          seq_length = input_size[1];
          mini_batch = input_size[0];
        } else {
          seq_length = input_size[0];
          mini_batch = input_size[1];
        }
        inner_size = input_size[2];
      }
    }

    // TODO: check x for consistency with input_size?
    std::vector<TensorDescriptor> descriptors(Tensor x) const {
      auto is_input_packed = batch_sizes.size() != 0;
      if (is_input_packed) {
        return rnn_descriptor_sequence(x, batch_sizes);
      } else {
        return rnn_descriptor(x[0], seq_length);
      }
    }
  };

  // Everything together

  struct RNNParams {
    DropoutDescriptorParams dropout;
    RNNDescriptorParams rnn;
    TensorDescriptorListParams tensors;
  };

  // NB: Doesn't include the weight descriptor
  struct RNNDescriptors {
    RNNDescriptor rnn_desc;
    // NB: this won't actually lay out the tensor descriptor pointers
    // in the right way, so you'll have to preprocess them
    std::vector<TensorDescriptor> x_descs;
    std::vector<TensorDescriptor> y_descs;
    TensorDescriptor hx_desc;
    TensorDescriptor hy_desc;
    TensorDescriptor cx_desc;
    TensorDescriptor cy_desc;

    RNNDescriptors(const RNNParams& fn, cudnnHandle_t handle, Tensor x, Tensor y, Tensor hx, Tensor cx) {
      rnn_desc = fn.rnn.descriptor(handle, fn.dropout.descriptor(handle));
      x_descs = fn.tensors.descriptors(x);
      y_descs = fn.tensors.descriptors(y);
      hx_desc.set(hx, 5);
      hy_desc.set(hx, 5);
      if (cx.defined()) {
        cx_desc.set(cx, 5);
        cy_desc.set(cx, 5);
      }
    }

    // TODO: This is annoying, having to put the cudnnTensorDescriptor_t
    // in a contiguous array...
    std::vector<cudnnTensorDescriptor_t> get_descs(const std::vector<TensorDescriptor>& descs) {
      std::vector<cudnnTensorDescriptor_t> r;
      r.reserve(descs.size());
      for (auto& desc : descs) {
        r.emplace_back(desc.desc());
      }
      return r;
    }

    std::vector<cudnnTensorDescriptor_t> get_x_descs() {
      return get_descs(x_descs);
    }

    std::vector<cudnnTensorDescriptor_t> get_y_descs() {
      return get_descs(y_descs);
    }
  };

  int64_t get_num_weights(cudnnHandle_t handle, const RNNDescriptor& rnn_desc,
                          const TensorDescriptor& x_desc, cudnnDataType_t datatype) {
    size_t weight_size;
    CUDNN_CHECK(cudnnGetRNNParamsSize(handle, rnn_desc.desc(), x_desc.desc(), &weight_size, datatype));
    auto elem_size = dataSize(datatype);
    AT_ASSERT(weight_size % elem_size == 0, "cudnnGetRNNParamsSize returned nonsensical weight_size");
    return weight_size / elem_size;
  }

  int64_t _num_linear_layers(cudnnRNNMode_t mode) {
    switch(mode) {
      case CUDNN_LSTM:
        return 8;
      case CUDNN_GRU:
        return 6;
      case CUDNN_RNN_RELU:
        return 2;
      case CUDNN_RNN_TANH:
        return 2;
      default:
        at::runtime_error("unknown cuDNN RNN mode %d", mode);
    }
  }

  /*
    Returns weight and bias tensors for each layer of the RNN. These tensors
    are views on the underlying weight buffer allocated by CuDNN.

    Note: for LSTM and GRU, which have multiple parameters of each type (4 and 3, respectively),
          these parameters are concatenated along the first dimension.
          These parameters are returned in a consistent order by CuDNN:
              (reset, forget, cell, output) for LSTM
              (reset, input, new) for GRU
    Args:
        fn: The RNN function object holding the RNN state
        handle: a CuDNN handle
        weight_buf: a 1D tensor containing the CuDNN-allocated weight (or grad_weight) buffer
    Returns:
        parameters: [(weight_ih, weight_hh, bias_ih, bias_hh)*], with length equal to the num_layers.
            This is represented as a pair of vector, and outer-dimension stride
            (NB: Can't return MatrixRef because we need to allocate the underlying tensor)
  */
  std::pair<std::vector<Tensor>, size_t> // stride0
  get_parameters(
      cudnnHandle_t handle,
      const RNNDescriptorParams& rnn,
      const RNNDescriptor& rnn_desc,
      const TensorDescriptor& x_desc,
      const FilterDescriptor& w_desc,
      const Tensor& weight_buf
  ) {
    auto cudnn_methods = { cudnnGetRNNLinLayerMatrixParams, cudnnGetRNNLinLayerBiasParams };
    std::vector<Tensor> params;
    int64_t num_linear_layers = _num_linear_layers(rnn.mode);
    int64_t num_layers = rnn.num_directions() * rnn.num_layers;
    size_t cur_offset = 0;
    size_t global_layer_params_count = 0;
    for (int64_t layer = 0; layer < num_layers; layer++) {
      size_t layer_params_count = 0;
      for (auto cudnn_method : cudnn_methods) {
        for (int64_t linear_id = 0; linear_id < num_linear_layers; linear_id++) {
          FilterDescriptor lin_layer_mat_desc;
          void* matrix_pointer;
          CUDNN_CHECK(cudnn_method(
                handle,
                rnn_desc.desc(),
                layer,
                x_desc.desc(),
                w_desc.desc(),
                weight_buf.data_ptr(),
                linear_id,
                lin_layer_mat_desc.mut_desc(),
                &matrix_pointer
                ));
          cudnnDataType_t data_type;
          cudnnTensorFormat_t format;
          int nb_dims;
          constexpr int min_dim = 3;
          // TODO: The use of CPU tensor here is a bit goofy in C++,
          // some sort of alloca would be good enough except that it is
          // kind of convenient to be able to prod() on it.
          Tensor filter_dim_a = at::CPU(kInt).tensor(min_dim);
          CUDNN_CHECK(cudnnGetFilterNdDescriptor(
                lin_layer_mat_desc.desc(),
                min_dim,
                &data_type,
                &format,
                &nb_dims,
                filter_dim_a.data<int>()
                ));

          AT_ASSERT(nb_dims <= min_dim, "cudnnGetFilterNdDescriptor failed nb_dims (%d) <= min_dim (%d)", nb_dims, min_dim);
          filter_dim_a = filter_dim_a.slice(0, 0, nb_dims);
          auto elem_size = dataSize(rnn.datatype);
          auto offset_bytes = (char*)matrix_pointer - (char*)weight_buf.data_ptr();
          AT_ASSERT(offset_bytes % elem_size == 0, "offset_bytes `mod` elem_size != 0 (%d %% %d)", offset_bytes, elem_size);
          size_t offset = offset_bytes / elem_size;

          // for all the RNN types provided by CUDNN, all the ih weights
          // are the same size and are allocated in a contiguous chunk
          // (same for the hh weights, and the ih and hh biases).
          // Since we're storing all the weights in a single tensor anyway,
          // might as well merge the CUDNN ones into a single tensor as well
          if (linear_id == 0 || linear_id == num_linear_layers / 2) {
            AT_ASSERT(*filter_dim_a.prod().data<int>() == *filter_dim_a[0].data<int>(), "filter_dim_a.prod() == filter_dim_a[0]");
            std::initializer_list<int64_t> size = {
              *filter_dim_a[0].data<int>() * num_linear_layers / 2,
              *filter_dim_a[2].data<int>()};
            Tensor param = weight_buf.type().tensor().set_(*weight_buf.storage(), offset, size);
            params.emplace_back(std::move(param));
            layer_params_count++;
          } else {
            AT_ASSERT(cur_offset == offset, "cur_offset == offset");
          }
          cur_offset = offset + *filter_dim_a[0].data<int>();
        }
      } // for cudnn_method
      if (layer == 0) {
        global_layer_params_count = layer_params_count;
      } else {
        AT_ASSERT(global_layer_params_count == layer_params_count, "%d (global) != %d", global_layer_params_count, layer_params_count);
      }
    } // for layer
    return std::make_pair(params, global_layer_params_count);
  }

  void _copyParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to) {
    AT_ASSERT(params_from.size(0) == params_to.size(0), "number of layers mismatch");
    for (size_t i = 0; i < params_from.size(0); i++) {
      auto layer_params_from = params_from[i];
      auto layer_params_to = params_to[i];
      for (auto a = layer_params_from.begin(), b = layer_params_to.begin();
           a != layer_params_from.end() && b != layer_params_to.end();
           ++a, ++b) {
        auto param_from = *a, param_to = *b;
        AT_ASSERT(param_from.type() == param_to.type(), "parameter types mismatch");
        param_to.copy_(param_from.view_as(param_to));
      }
    }
  }

  std::vector<int64_t> _input_size(const TensorDescriptorListParams& tensors) {
    if (tensors.is_input_packed()) {
      return {tensors.outer_size, tensors.inner_size};
    } else {
      return {tensors.seq_length, tensors.mini_batch, tensors.inner_size};
    }
  }

  std::vector<int64_t> _hidden_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
    return {rnn.num_layers * rnn.num_directions(), tensors.mini_batch, rnn.hidden_size};
  }

  std::vector<int64_t> _output_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
    if (tensors.is_input_packed()) {
      return {tensors.outer_size, rnn.hidden_size * rnn.num_directions()};
    } else {
      return {tensors.seq_length, tensors.mini_batch, rnn.hidden_size * rnn.num_directions()};
    }
  }

} // anonymous namespace

// NB: does inplace update into TensorList
// It would be a relatively simple matter to refactor this into multiple
// functions, only one of which does an inplace update, but we leave this
// for future work
Tensor _cudnn_rnn_flatten_weight(
    TensorList weight_arr, int64_t weight_stride0,
    int64_t input_size,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first,
    bool fn_bidirectional
    ) {

  if (weight_arr.size() == 0) {
    throw std::runtime_error("_cudnn_rnn_flatten_weight_: cannot flatten empty weight list");
  }

  auto any_param = weight_arr[0];

  RNNDescriptorParams rnn;
  rnn.set_mode(fn_mode);
  rnn.hidden_size = fn_hidden_size;
  rnn.num_layers = fn_num_layers;
  rnn.set_bidirectional(fn_bidirectional);
  rnn.datatype = getCudnnDataType(any_param);

  auto handle = getCudnnHandle();
  // NB: So, I am pretty sure that get_parameters() does not rely in any way
  // on the dropout descriptor.  So we fake up a dummy one instead of try
  // to actually make one legitimately.
  RNNDescriptor rnn_desc = rnn.descriptor(handle);

  // TODO: allocation here is goofy
  TensorDescriptor x_desc(any_param.type().tensor({1, input_size}), 5);

  auto num_weights = get_num_weights(handle, rnn_desc, x_desc, rnn.datatype);
  auto weight_buf = any_param.type().tensor(num_weights).zero_();

  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);

  // Slice off views into weight_buf
  std::vector<Tensor> params_arr;
  size_t params_stride0;
  std::tie(params_arr, params_stride0) = get_parameters(handle, rnn, rnn_desc, x_desc, w_desc, weight_buf);

  MatrixRef<Tensor> weight{weight_arr, static_cast<size_t>(weight_stride0)},
                    params{params_arr, params_stride0};

  // Copy weights
  _copyParams(weight, params);

  // Update the storage
  for (size_t i = 0; i < weight.size(0); i++) {
    for (auto orig_param_it = weight[i].begin(), new_param_it = params[i].begin();
         orig_param_it != weight[i].end() && new_param_it != params[i].end();
         orig_param_it++, new_param_it++) {
      auto orig_param = *orig_param_it, new_param = *new_param_it;
      orig_param.set_(new_param.view_as(orig_param));
    }
  }

  return weight_buf;
}

// NB: when fn_batch_sizes is empty, that means no batch sizes was specified
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _cudnn_rnn(
    const Tensor& input_r,
    TensorList weight, int64_t weight_stride0,
    const Tensor& weight_buf_r, const Tensor& hx, const Tensor& cx,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntList fn_batch_sizes,
    const Tensor& fn_dropout_state
    ) {

  auto input = input_r;
  auto weight_buf = weight_buf_r;

  RNNParams fn;
  fn.rnn.set_mode(fn_mode);
  fn.rnn.hidden_size = fn_hidden_size;
  fn.rnn.num_layers = fn_num_layers;
  fn.rnn.set_bidirectional(fn_bidirectional);
  fn.rnn.datatype = getCudnnDataType(input);
  fn.dropout.set(fn_train, fn_dropout, fn_dropout_state);
  fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

  // TODO: Set device to input

  if (fn.rnn.mode != CUDNN_LSTM) {
    if (cx.defined()) {
      throw std::runtime_error("rnn: illegal defined cx for non-LSTM RNN");
    }
  }

  // TODO: can batch_first be a wrapper around this function?
  auto is_input_packed = fn.tensors.batch_sizes.size() != 0;
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
  }

  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
  auto output_size = _output_size(fn.rnn, fn.tensors);

  AT_ASSERT(hx.is_contiguous(), "hx.is_contiguous()");
  AT_ASSERT(!cx.defined() || cx.is_contiguous(), "!cx or cx.is_contiguous()");

  auto x = input.contiguous();
  auto output = input.type().tensor(output_size);
  auto hy = hx.type().tensor(hidden_size);
  Tensor cy;
  if (cx.defined()) {
    cy = cx.type().tensor(hidden_size);
  } else {
    cy = hx.type().tensor(); // NB: Not allowed to return undefined tensors
  }
  auto y = output;

  auto handle = getCudnnHandle();
  RNNDescriptors descs(fn, handle, x, y, hx, cx);

  FilterDescriptor w_desc;
  if (!weight_buf.defined()) {
    auto num_weights = get_num_weights(handle, descs.rnn_desc, descs.x_descs[0], fn.rnn.datatype);
    weight_buf = x.type().tensor(num_weights);
    w_desc.set(weight_buf, 3);
    weight_buf.zero_();
    std::vector<Tensor> params;
    size_t params_stride0;
    std::tie(params, params_stride0) = get_parameters(handle, fn.rnn, descs.rnn_desc, descs.x_descs[0], w_desc, weight_buf);
    _copyParams(MatrixRef<Tensor>{weight, static_cast<size_t>(weight_stride0)},
                MatrixRef<Tensor>{params, params_stride0});
  } else {
    w_desc.set(weight_buf, 3);
  }

  if (cx.defined() && !cx.sizes().equals(hidden_size)) {
    std::ostringstream oss;
    oss << "Expected cell size " << IntList{hidden_size} << ", got " << cx.sizes();
    throw std::runtime_error(oss.str());
  }

  size_t workspace_size;
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(),
        &workspace_size
        ));
  Tensor workspace = input.type().toScalarType(kByte).tensor(workspace_size);

  Tensor reserve;
  // NB: Previously, the test was for fn.requires_grad, but we don't have
  // this information.  Use 'train' as a proxy.
  if (fn.dropout.train) {
    size_t reserve_size;
    CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
          handle,
          descs.rnn_desc.desc(),
          fn.tensors.seq_length,
          x_descs_arr.data(),
          &reserve_size
          ));
    reserve = input.type().toScalarType(kByte).tensor(reserve_size);
    CUDNN_CHECK(cudnnRNNForwardTraining(
          handle,
          descs.rnn_desc.desc(),
          fn.tensors.seq_length,
          x_descs_arr.data(), x.data_ptr(),
          descs.hx_desc.desc(), hx.data_ptr(),
          descs.cx_desc.desc(), cx.defined() ? cx.data_ptr() : nullptr,
          w_desc.desc(), weight_buf.data_ptr(),
          y_descs_arr.data(), y.data_ptr(),
          descs.hy_desc.desc(), hy.data_ptr(),
          descs.cy_desc.desc(), cy.defined() ? cy.data_ptr() : nullptr,
          workspace.data_ptr(), workspace.size(0),
          reserve.data_ptr(), reserve.size(0)
          ));
  } else { // inference
    reserve = input.type().toScalarType(kByte).tensor();
    CUDNN_CHECK(cudnnRNNForwardInference(
          handle,
          descs.rnn_desc.desc(),
          fn.tensors.seq_length,
          x_descs_arr.data(), x.data_ptr(),
          descs.hx_desc.desc(), hx.data_ptr(),
          descs.cx_desc.desc(), cx.defined() ? cx.data_ptr() : nullptr,
          w_desc.desc(), weight_buf.data_ptr(),
          y_descs_arr.data(), y.data_ptr(),
          descs.hy_desc.desc(), hy.data_ptr(),
          descs.cy_desc.desc(), cy.defined() ? cy.data_ptr() : nullptr,
          workspace.data_ptr(), workspace.size(0)
          ));

  }

  if (batch_first && !is_input_packed) {
    output.transpose_(0, 1);
  }

  return std::make_tuple(output, hy, cy, reserve, weight_buf);
}

std::tuple<Tensor, Tensor, Tensor> _cudnn_rnn_backward_grad(
    const Tensor& input_r, const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
    const Tensor& output_r, const Tensor& grad_output_r, const Tensor& grad_hy,
    const Tensor& grad_cy,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntList fn_batch_sizes,
    const Tensor& fn_dropout_state, const Tensor& fn_reserve
    ) {

  auto input = input_r;
  auto grad_output = grad_output_r;
  auto output = output_r;

  RNNParams fn;
  fn.rnn.set_mode(fn_mode);
  fn.rnn.hidden_size = fn_hidden_size;
  fn.rnn.num_layers = fn_num_layers;
  fn.rnn.set_bidirectional(fn_bidirectional);
  fn.rnn.datatype = getCudnnDataType(input);
  fn.dropout.set(fn_train, fn_dropout, fn_dropout_state);
  fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

  // TODO: Set device to input
  auto handle = getCudnnHandle();

  if (fn.rnn.mode != CUDNN_LSTM) {
    if (cx.defined()) {
      throw std::runtime_error("rnn: illegal defined cx for non-LSTM RNN");
    }
  }

  auto is_input_packed = fn_batch_sizes.size() != 0;
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
    grad_output = grad_output.transpose(0, 1);
    output = output.transpose(0, 1);
  }

  auto input_size = _input_size(fn.tensors);
  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
  auto output_size = _output_size(fn.rnn, fn.tensors);

  AT_ASSERT(hx.is_contiguous(), "hx.is_contiguous()");
  AT_ASSERT(!cx.defined() || cx.is_contiguous(), "!cx or cx.is_contiguous()");

  auto x = input.contiguous();
  auto dy = grad_output.contiguous();
  auto y = output;
  auto w = weight_buf;
  auto dx = input.type().tensor(input.sizes()); // TODO: more compact way of saying this
  auto dhy = grad_hy.contiguous().view(hidden_size);
  auto dcy = grad_cy.defined() ? grad_cy.contiguous().view(hidden_size) : Tensor();
  auto dhx = hx.type().tensor(hidden_size);
  auto dcx = cx.defined() ? cx.type().tensor(hidden_size) : hx.type().tensor(); // Boooh

  if (!fn.dropout.train) {
    throw std::runtime_error("backward_grad can only be called in training mode");
  }
  if (!input.sizes().equals(input_size)) {
    std::ostringstream oss;
    oss << "Expected input size " << IntList{input_size} << ", got " << input.sizes();
    throw std::runtime_error(oss.str());
  }
  if (!output.sizes().equals(output_size)) {
    std::ostringstream oss;
    oss << "Expected output size " << IntList{output_size} << ", got " << output.sizes();
    throw std::runtime_error(oss.str());
  }
  if (hx.defined() && !hx.sizes().equals(hidden_size)) {
    std::ostringstream oss;
    oss << "Expected hidden size " << IntList{hidden_size} << ", got " << hx.sizes();
    throw std::runtime_error(oss.str());
  }
  if (cx.defined() && !cx.sizes().equals(hidden_size)) {
    std::ostringstream oss;
    oss << "Expected cell size " << IntList{hidden_size} << ", got " << cx.sizes();
    throw std::runtime_error(oss.str());
  }
  if (dhy.defined() && !dhy.sizes().equals(hidden_size)) {
    std::ostringstream oss;
    oss << "Expected d_hidden size " << IntList{hidden_size} << ", got " << dhy.sizes();
    throw std::runtime_error(oss.str());
  }
  if (dcy.defined() && !dcy.sizes().equals(hidden_size)) {
    std::ostringstream oss;
    oss << "Expected d_cell size " << IntList{hidden_size} << ", got " << dcy.sizes();
    throw std::runtime_error(oss.str());
  }
  if (!dhy.is_cuda() || !dy.is_cuda() || (dcy.defined() && !dcy.is_cuda())) {
    throw std::runtime_error("Gradients aren't CUDA tensors");
  }

  RNNDescriptors descs(fn, handle, x, y, hx, cx);

  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);

  size_t workspace_size;
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(),
        &workspace_size
        ));
  // TODO: put this in the correct device???
  Tensor workspace = input.type().toScalarType(kByte).tensor(workspace_size);

  CUDNN_CHECK(cudnnRNNBackwardData(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        y_descs_arr.data(), y.data_ptr(),
        y_descs_arr.data(), dy.data_ptr(),
        descs.hy_desc.desc(), dhy.data_ptr(),
        descs.cy_desc.desc(), cx.defined() ? dcy.data_ptr() : nullptr,
        w_desc.desc(), w.data_ptr(),
        descs.hx_desc.desc(), hx.data_ptr(),
        descs.cx_desc.desc(), cx.defined() ? cx.data_ptr() : nullptr,
        x_descs_arr.data(), dx.data_ptr(),
        descs.hx_desc.desc(), dhx.data_ptr(),
        descs.cx_desc.desc(), cx.defined() ? dcx.data_ptr() : nullptr,
        workspace.data_ptr(), workspace.size(0),
        fn_reserve.data_ptr(), fn_reserve.size(0)
        ));

  if (batch_first && !is_input_packed) {
    dx = dx.transpose_(0, 1);
  }

  return std::make_tuple(dx, dhx, dcx); // TODO
}

// NB: This MUST BE CALLED AFTER _cudnn_rnn_backward_grad.
// We'll give a user friendly combined function...
std::vector<Tensor> _cudnn_rnn_backward_weight(
    // TODO: I think tensor geometry sufficient for weight_buf/weight
    const Tensor& input_r, TensorList weight_arr, int64_t weight_stride0,
    const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
    const Tensor& output_r,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntList fn_batch_sizes,
    const Tensor& fn_dropout_state, const Tensor& fn_reserve
    ) {

  MatrixRef<Tensor> weight{ weight_arr, static_cast<size_t>(weight_stride0) };

  auto input = input_r;
  auto output = output_r;

  RNNParams fn;
  fn.rnn.set_mode(fn_mode);
  fn.rnn.hidden_size = fn_hidden_size;
  fn.rnn.num_layers = fn_num_layers;
  fn.rnn.set_bidirectional(fn_bidirectional);
  fn.rnn.datatype = getCudnnDataType(input);
  fn.dropout.set(fn_train, fn_dropout, fn_dropout_state);
  fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

  auto handle = getCudnnHandle();

  if (fn.rnn.mode != CUDNN_LSTM) {
    if (cx.defined()) {
      throw std::runtime_error("rnn: illegal defined cx for non-LSTM RNN");
    }
  }

  auto is_input_packed = fn_batch_sizes.size() != 0;
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
    output = output.transpose(0, 1);
  }

  auto input_size = _input_size(fn.tensors);
  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);

  if (!fn.dropout.train) {
    throw std::runtime_error("backward_grad can only be called in training mode");
  }
  if (!input.sizes().equals(input_size)) {
    std::ostringstream oss;
    oss << "Expected input size " << IntList{input_size} << ", got " << input.sizes();
    throw std::runtime_error(oss.str());
  }
  if (hx.defined() && !hx.sizes().equals(hidden_size)) {
    std::ostringstream oss;
    oss << "Expected hidden size " << IntList{hidden_size} << ", got " << hx.sizes();
    throw std::runtime_error(oss.str());
  }
  // TODO: the above were the only checks in rnn.py, but it doesn't seem
  // like these checks are enough

  AT_ASSERT(hx.is_contiguous(), "hx.is_contiguous()");
  AT_ASSERT(!cx.defined() || cx.is_contiguous(), "!cx or cx.is_contiguous()");

  auto x = input.contiguous();
  const auto& y = output;
  auto dw = weight_buf.type().tensor(weight_buf.sizes()).zero_();

  RNNDescriptors descs(fn, handle, x, y, hx, cx);

  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);

  size_t workspace_size;
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(),
        &workspace_size
        ));
  Tensor workspace = input.type().toScalarType(kByte).tensor(workspace_size);

  CUDNN_CHECK(cudnnRNNBackwardWeights(
        handle,
        descs.rnn_desc.desc(),
        fn.tensors.seq_length,
        x_descs_arr.data(), x.data_ptr(),
        descs.hx_desc.desc(), hx.data_ptr(),
        y_descs_arr.data(), y.data_ptr(),
        workspace.data_ptr(), workspace.size(0),
        w_desc.desc(), dw.data_ptr(),
        fn_reserve.data_ptr(), fn_reserve.size(0)
        ));

  std::vector<Tensor> grad_weight_arr;
  grad_weight_arr.reserve( weight.numel() );
  for (const auto& w : weight_arr) {
    grad_weight_arr.emplace_back(w.type().tensor(w.sizes()).zero_());
  }

  std::vector<Tensor> grad_params_arr;
  size_t grad_params_stride0;
  std::tie(grad_params_arr, grad_params_stride0) = get_parameters(handle, fn.rnn, descs.rnn_desc, descs.x_descs[0], w_desc, dw);
  _copyParams(MatrixRef<Tensor>{grad_params_arr, grad_params_stride0},
              MatrixRef<Tensor>{grad_weight_arr, static_cast<size_t>(weight_stride0)});

  return grad_weight_arr; // stride is known from call site (and also inconvenient to return)
}

// We need this dispatcher because _cudnn_rnn_backward_weight has a stringent
// ordering requirement with _cudnn_rnn_backward_grad
std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> _cudnn_rnn_backward(
    const Tensor& input, TensorList weight, int64_t weight_stride0, const Tensor& weight_buf, const Tensor& hx, const Tensor& cx,
    const Tensor& output, const Tensor& grad_output_r, const Tensor& grad_hy_r,
    const Tensor& grad_cy_r,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool batch_first, double dropout,
    bool train, bool bidirectional, IntList batch_sizes,
    const Tensor& dropout_state, const Tensor& reserve,
    std::array<bool, 4> output_mask
    ) {

  auto grad_output = grad_output_r.defined() ? grad_output_r : output.type().zeros_like(output);
  auto grad_hy = grad_hy_r.defined() ? grad_hy_r : hx.type().zeros_like(hx);
  auto grad_cy = cx.defined() ? (grad_cy_r.defined() ? grad_cy_r : cx.type().zeros_like(cx)) : grad_cy_r;

  Tensor dx, dhx, dcx;
  // NB: unconditionally compute this gradient, because it mutates reserve
  std::tie(dx, dhx, dcx) = at::native::_cudnn_rnn_backward_grad(input, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve);
  std::vector<Tensor> dw;
  if (output_mask[3]) {
    dw = at::native::_cudnn_rnn_backward_weight(input, weight, weight_stride0, weight_buf, hx, cx, output, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve);
  }
  return std::tuple<Tensor, Tensor, Tensor, TensorList>{dx, dhx, dcx, dw};
}

}} // namespace at::native

#endif // AT_CUDNN_ENABLED()
