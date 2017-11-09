#include "Conv.h"

#include "THC/THC.h"
#include "Exceptions.h"
#include "Utils.h"
#include "Types.h"

#include <ATen/Check.h>

#include "cudnn-wrapper.h"
#include <functional>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>

namespace at { namespace cudnn {

// TODO: Go through all the checking code again and make sure
// we haven't missed anything.

// ---------------------------------------------------------------------
//
// Math
//
// ---------------------------------------------------------------------

constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int input_channels_dim = 1;
constexpr int output_batch_size_dim = 0;  // also grad_output
constexpr int output_channels_dim = 1;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

// Often written as 2 + max_dim (extra dims for batch size and channels)
constexpr int max_dim = 3;

// NB: conv_output_size and conv_input_size are not bijections,
// as conv_output_size loses information; this is why conv_input_size
// takes an extra output_padding argument to resolve the ambiguity.

std::vector<int64_t> conv_output_size(
    IntList input_size, IntList weight_size,
    IntList padding, IntList stride, IntList dilation, int groups
) {
  // ASSERT(input_size.size() > 2)
  // ASSERT(input_size.size() == weight_size.size())
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (int d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2])
                        - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

std::vector<int64_t> conv_input_size(
    IntList output_size, IntList weight_size,
    IntList padding, IntList output_padding, IntList stride, IntList dilation, int groups
) {
  // ASSERT(output_size.size() > 2)
  // ASSERT(output_size.size() == weight_size.size())
  auto dim = output_size.size();
  std::vector<int64_t> input_size(dim);
  input_size[0] = output_size[output_batch_size_dim];
  input_size[1] = weight_size[weight_input_channels_dim] * groups;
  for (int d = 2; d < dim; ++d) {
    int kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 2] - (2 * padding[d - 2]) +
                     kernel + output_padding[d - 2];
  }
  return input_size;
}

std::vector<int64_t> conv_weight_size(
    IntList input_size, IntList output_size,
    IntList padding, IntList output_padding, IntList stride, IntList dilation, int groups
) {
  auto dim = input_size.size();
  std::vector<int64_t> weight_size(dim);
  weight_size[0] = output_size[1];
  weight_size[1] = input_size[1] / groups;
  for (int d = 2; d < dim; ++d) {
    int kernel = input_size[d] - (output_size[d] - 1) * stride[d - 2]
               + 2 * padding[d - 2] - output_padding[d - 2];
    weight_size[d] = (kernel - 1) / dilation[d - 2] + 1;
  }
  return weight_size;
}

// TODO: Move this into the standard library, with a better name?
Tensor narrowGroup(const Tensor& t, int dim, int group_idx, int groups) {
  auto group_size = t.size(dim) / groups;
  return t.narrow(dim, group_idx * group_size, group_size);
}

// ---------------------------------------------------------------------
//
// Checking
//
// ---------------------------------------------------------------------

// Note [Legacy CuDNN grouped convolution support]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CuDNN earlier than CuDNN 7 does not directly support group
// convolution, so we provide support for it by sequentially
// running a convolution per group  with appropriately
// adjusted sizes.  https://blog.yani.io/filter-group-tutorial/
// has a fairly good diagram explaining how it works.

// Used on pad, stride and dilation
static void check_args(CheckedFrom c, IntList args, size_t expected_size, const char* arg_name)
{
  if (args.size() > expected_size){
    std::stringstream ss;
    ss << "Too many " << arg_name << " values (" << args.size() << ") supplied, expecting " << expected_size << " (while checking arguments for " << c << ")";
    throw std::runtime_error(ss.str());
  }
  else if (args.size() < expected_size){
    std::stringstream ss;
    ss << "Not enough " << arg_name << " values (" << args.size() << ") supplied, expecting " << expected_size << " (while checking arguments for " << c << ")";
    throw std::runtime_error(ss.str());
  }

  auto num_negative_values = std::count_if(args.begin(), args.end(), [](int x){return x < 0;});
  if (num_negative_values > 0){
    std::stringstream ss;
    ss << arg_name << " should be greater than zero but got (";
    std::copy(args.begin(), args.end() - 1, std::ostream_iterator<int>(ss,", "));
    ss << args.back() <<  ")" << " (while checking arguments for " << c << ")";
    throw std::runtime_error(ss.str());
  }
}


// NB: For many call sites, it is not strictly necessary to check all of
// these relationships (for example, for forward convolution, we compute
// the size of output ourselves, so we don't actually need to check
// output.  However, writing a single function that does everything
// means we get to reuse it for both forwards and all backwards
// variants, even when the set of "real" inputs varies.  The magic of
// relational computing!
//
// (There is one downside, which is that it is slightly harder to write
// error messages which are able to distinguish between real inputs
// (which the user can change) and computed inputs (which the user can
// only indirectly affect).  It would be an interesting exercise to
// come up with a general framework to handle such situations.)
static void convolution_shape_check(
    CheckedFrom c,
    TensorGeometryArg input, TensorGeometryArg weight, TensorGeometryArg output,
    IntList padding, IntList stride, IntList dilation, int groups)
{
  check_args(c, padding, input->dim() - 2, "padding");
  check_args(c, stride, padding.size(), "stride");
  check_args(c, dilation, padding.size(), "dilation");

  // Input
  checkDimRange(c, input, 3, 6 /* exclusive */);
  checkSize(c, input, input_channels_dim, weight->size(1) * groups);

  // Weight
  checkContiguous(c, weight);
  checkSameDim(c, input, weight);

  // Output
  {
    auto output_sizes = conv_output_size(input->sizes(), weight->sizes(),
                                         padding, stride, dilation, groups);
    bool invalid_dim_size = false;
    for (auto output : output_sizes) {
      if (output < 1) invalid_dim_size = true;
    }
    if (invalid_dim_size){
      // NB: This message only makes sense for convolution (not
      // transposed convolution); however, you'd be hard pressed to
      // trigger it in a transposed convolution, as it would imply that
      // you had passed in tensor with zero-size dimensions.
      std::stringstream ss;
      ss <<  "Given input size " << input->sizes()
         << ", calculated output size" << IntList(output_sizes)
         << ", but got too small size for " << output
         << " (while checking arguments for " << c << ")";
      throw std::runtime_error(ss.str());
    }
  }
  // TODO: check that output->size() matches output_sizes
  // TODO: check that weight matches output->sizes()
  checkSameDim(c, input, output);
}

// This POD struct is used to let us easily compute hashes of the
// parameters
struct ConvolutionParams
{
  cudnnDataType_t dataType;
  int input_size[2 + max_dim];
  int input_stride[2 + max_dim];
  int weight_size[2 + max_dim];
  int padding[max_dim];
  int stride[max_dim];
  int dilation[max_dim];
  int groups;
  bool deterministic;
  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
};
// ConvolutionParams must be a POD because we read out its memory
// contenst as char* when hashing
static_assert(std::is_pod<ConvolutionParams>::value, "ConvolutionParams not POD");

// NB: This can't be a constructor, because then ConvolutionParams
// would not be a POD anymore.
// TODO: Use TensorGeometry here instead of the entire Tensor, which we
// don't actually need.  (OTOH: We can always pass in
// grad_input/grad_output, so this is not very pressing)
void setConvolutionParams(
    ConvolutionParams* params,
    const at::Tensor& input, const at::Tensor& weight,
    IntList padding, IntList stride, IntList dilation,
    int groups, bool deterministic) {

  cudnnDataType_t dataType = getCudnnDataType(input);
  memset(params, 0, sizeof(ConvolutionParams));
  params->dataType = dataType;
  // ASSERT(weight.dim() == input.dim())
  for (int i = 0; i != input.dim(); ++i) {
    params->input_size[i] = (int) input.size(i);
    params->input_stride[i] = (int) input.stride(i);
    params->weight_size[i] = (int) weight.size(i);
  }
  // ASSERT(padding.size() == stride.size())
  // ASSERT(padding.size() == dilation.size())
  for (size_t i = 0; i != padding.size(); ++i) {
    params->padding[i] = padding[i];
    params->stride[i] = stride[i];
    params->dilation[i] = dilation[i];
  }
  // In principle, we shouldn't parametrize by groups for legacy
  // CuDNN, but it doesn't seem worth the effort to actually do this.
  params->groups = groups;
  params->deterministic = deterministic;
}

// ---------------------------------------------------------------------
//
// Benchmarking
//
// ---------------------------------------------------------------------

// Hashing machinery for ConvolutionParams
struct ParamsHash {
  std::size_t operator()(const ConvolutionParams& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    for (int i = 0; i < (int)sizeof(ConvolutionParams); ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

struct ParamsEqual {
  bool operator()(const ConvolutionParams& a, const ConvolutionParams& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    return memcmp(ptr1, ptr2, sizeof(ConvolutionParams)) == 0;
  }
};

// TODO: Use something less heavy duty than a big honking mutex
template <typename T>
struct BenchmarkCache {
  std::mutex mutex;
  std::unordered_map<ConvolutionParams, T, ParamsHash, ParamsEqual> map;

  bool find(const ConvolutionParams& params, T* results) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    *results = it->second;
    return true;
  }

  void insert(const ConvolutionParams& params, const T& results) {
    std::lock_guard<std::mutex> guard(mutex);
    map[params] = results;
  }
};

BenchmarkCache<cudnnConvolutionFwdAlgo_t> fwd_algos;
BenchmarkCache<cudnnConvolutionBwdDataAlgo_t> bwd_data_algos;
BenchmarkCache<cudnnConvolutionBwdFilterAlgo_t> bwd_filter_algos;

// TODO: Stop manually allocating CUDA memory; go through ATen
// (ATen should provide an API for this!)
struct Workspace {
  Workspace(size_t size) : size(size), data(NULL) {
    CUDA_CHECK(THCudaMalloc(globalContext().lazyInitCUDA(), &data, size));
  }
  Workspace(const Workspace&) = delete;
  Workspace(Workspace&&) = default;
  ~Workspace() {
    if (data) {
      THCudaFree(globalContext().lazyInitCUDA(), data);
    }
  }

  size_t size;
  void* data;
};

template<typename algo_t>
struct algorithm_search {
};

cudnnStatus_t getWorkspaceSize(
    cudnnHandle_t handle,
    const ConvolutionDescriptor& cdesc,
    const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc,
    cudnnConvolutionFwdAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        idesc.desc,
        wdesc.desc,
        cdesc.desc,
        odesc.desc,
        algo,
        sz
    );
}
cudnnStatus_t getWorkspaceSize(
    cudnnHandle_t handle,
    const ConvolutionDescriptor& cdesc,
    const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle,
        wdesc.desc,
        odesc.desc,
        cdesc.desc,
        idesc.desc,
        algo,
        sz);
}
cudnnStatus_t getWorkspaceSize(
    cudnnHandle_t handle,
    const ConvolutionDescriptor& cdesc,
    const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle,
        idesc.desc,
        odesc.desc,
        cdesc.desc,
        wdesc.desc,
        algo,
        sz);
}

template<typename algo_t>
size_t getMaxWorkspaceSize(
    cudnnHandle_t handle,
    const ConvolutionDescriptor& cdesc,
    const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc,
    const algo_t *algo, int n_algo)
{
    THCState *state = globalContext().lazyInitCUDA();

    size_t max_ws_size = 0;
    size_t max_block_size = 0;
    size_t total_gpu_mem = 0;
    size_t free_gpu_mem = 0;

    THCudaCheck(THCudaMemGetInfoCached(state, &free_gpu_mem, &total_gpu_mem, &max_block_size));

    for (int i = 0; i < n_algo; i++) {
        cudnnStatus_t err;
        size_t sz;
        err = getWorkspaceSize(handle, cdesc, idesc, odesc, wdesc, algo[i], &sz);
        if (CUDNN_STATUS_SUCCESS != err || sz == 0
            || sz < max_ws_size || sz > max_block_size) continue;
        max_ws_size = sz;
    }
    return max_ws_size;
}

template<typename perf_t>
perf_t getBestAlgorithm(perf_t *perfResults, bool deterministic, int n_algo) {
  if (deterministic) {
    // iterate over perf results of all algorithms and find the best deterministic algo
    for (int i = 0; i < n_algo; i++) {
      // TODO: Shouldn't all returned results be successful?
      // Double check documentation for cudnnFindConvolutionForwardAlgorithmEx
      if (perfResults[i].status == CUDNN_STATUS_SUCCESS &&
          perfResults[i].determinism == CUDNN_DETERMINISTIC) {
        return perfResults[i];
      }
    }
    throw std::runtime_error("no deterministic convolution algorithms available in CuDNN");
  } else {
    return perfResults[0];
  }
}

template<>
struct algorithm_search<cudnnConvolutionFwdAlgo_t> {
  using perf_t = cudnnConvolutionFwdAlgoPerf_t;
  using algo_t = cudnnConvolutionFwdAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  static BenchmarkCache<algo_t>& cache() { return fwd_algos; }

  static perf_t findAlgorithm(
      cudnnHandle_t handle,
      ConvolutionParams* params,
      const ConvolutionDescriptor& cdesc,
      const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc)
  {
    static const algo_t algos[] = {
         CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };
    static constexpr int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution forward algorithms");
    int perf_count;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    size_t max_ws_size = getMaxWorkspaceSize(handle, cdesc, idesc, odesc, wdesc, algos, num_algos);
    Workspace ws(max_ws_size);
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
        handle,
        idesc.desc, idesc.ptr,
        wdesc.desc, wdesc.ptr,
        cdesc.desc,
        odesc.desc, odesc.ptr,
        num_algos,
        &perf_count,
        perf_results.get(),
        ws.data,
        ws.size));
    return getBestAlgorithm(perf_results.get(), params->deterministic, perf_count);
  }

  static void getAlgorithm(
    cudnnHandle_t handle,
    const ConvolutionDescriptor& cdesc,
    const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc,
    algo_t* algo)
  {
    cudnnConvolutionFwdPreference_t pref = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
        handle,
        idesc.desc,
        wdesc.desc,
        cdesc.desc,
        odesc.desc,
        pref,
        0,
        algo));
  }

  static void getWorkspaceSize(
    cudnnHandle_t handle,
    const ConvolutionDescriptor& cdesc,
    const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc,
    algo_t algo, size_t* workspaceSize)
  {
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        idesc.desc,
        wdesc.desc,
        cdesc.desc,
        odesc.desc,
        algo,
        workspaceSize));
  }
};

template<>
struct algorithm_search<cudnnConvolutionBwdDataAlgo_t> {
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdDataAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  static BenchmarkCache<algo_t>& cache() { return bwd_data_algos; }

  static perf_t findAlgorithm(
        cudnnHandle_t handle,
        ConvolutionParams* params,
        const ConvolutionDescriptor& cdesc,
        const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc)
  {
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
    };
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward data algorithms.");
    int perf_count;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    size_t max_ws_size = getMaxWorkspaceSize(handle, cdesc, idesc, odesc, wdesc, algos, num_algos);
    Workspace ws(max_ws_size);
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithmEx(
        handle,
        wdesc.desc, wdesc.ptr,
        odesc.desc, odesc.ptr,
        cdesc.desc,
        idesc.desc, idesc.ptr,
        num_algos,
        &perf_count,
        perf_results.get(),
        ws.data,
        ws.size));
    return getBestAlgorithm(perf_results.get(), params->deterministic, perf_count);
  }

  static void getAlgorithm(cudnnHandle_t handle,
        const ConvolutionDescriptor& cdesc,
        const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc,
        algo_t* algo) {
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
        handle,
        wdesc.desc,
        odesc.desc,
        cdesc.desc,
        idesc.desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        algo));
  }

  static void getWorkspaceSize(
    cudnnHandle_t handle,
    const ConvolutionDescriptor& cdesc,
    const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* workspaceSize)
  {
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle,
        wdesc.desc,
        odesc.desc,
        cdesc.desc,
        idesc.desc,
        algo,
        workspaceSize));
  }
};

template<>
struct algorithm_search<cudnnConvolutionBwdFilterAlgo_t> {
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdFilterAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  static BenchmarkCache<algo_t>& cache() { return bwd_filter_algos; }

  static perf_t findAlgorithm(
        cudnnHandle_t handle, ConvolutionParams* params,
        const ConvolutionDescriptor& cdesc,
        const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc)
  {
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
#if CUDNN_VERSION >= 6000
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
#endif
    };
    // NOTE: - 1 because ALGO_WINOGRAD is not implemented
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT - 1;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward filter algorithms.");
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    size_t max_ws_size = getMaxWorkspaceSize(
        handle, cdesc, idesc, odesc, wdesc, algos, num_algos);
    int perf_count;
    Workspace ws(max_ws_size);

    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        handle,
        idesc.desc, idesc.ptr,
        odesc.desc, odesc.ptr,
        cdesc.desc,
        wdesc.desc, wdesc.ptr,
        num_algos,
        &perf_count,
        perf_results.get(),
        ws.data,
        ws.size));
    return getBestAlgorithm<perf_t>(perf_results.get(), params->deterministic, perf_count);
  }

  static void getAlgorithm(
      cudnnHandle_t handle,
      const ConvolutionDescriptor& cdesc,
      const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc,
      algo_t* algo)
  {
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
        handle,
        idesc.desc,
        odesc.desc,
        cdesc.desc,
        wdesc.desc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        0,
        algo)
    );
  }

  static void getWorkspaceSize(
      cudnnHandle_t handle,
      const ConvolutionDescriptor& cdesc,
      const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc,
      algo_t algo, size_t* workspaceSize)
  {
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle,
        idesc.desc,
        odesc.desc,
        cdesc.desc,
        wdesc.desc,
        algo,
        workspaceSize));
  }
};

template<typename algo_t>
void findAlgorithm(
    cudnnHandle_t handle, ConvolutionParams* params,
    bool benchmark,
    const ConvolutionDescriptor& cdesc,
    const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc,
    algo_t* algo)
{
  using search = algorithm_search<algo_t>;
  auto& cache = search::cache();

  if (cache.find(*params, algo)) {
    return;
  }

  if (params->deterministic && !benchmark) {
    *algo = search::DEFAULT_ALGO;
    return;
  }

  if (!benchmark) {
    search::getAlgorithm(handle, cdesc, idesc, odesc, wdesc, algo);
    return;
  }

  if (cache.find(*params, algo)) {
    // re-check cache since another thread may have benchmarked the algorithm
    return;
  }

  auto perfResults = search::findAlgorithm(handle, params, cdesc, idesc, odesc, wdesc);
  // for deterministic algo, look at all the perf results and return the best
  // deterministic algo
  if (perfResults.status == CUDNN_STATUS_SUCCESS &&
      !(params->deterministic && perfResults.determinism != CUDNN_DETERMINISTIC)) {
      *algo = perfResults.algo;
  } else {
      *algo = search::DEFAULT_ALGO;
  }
  cache.insert(*params, *algo);

  THCDeviceAllocator* allocator = THCCachingAllocator_get();
  CUDA_CHECK(allocator->emptyCache(allocator->state));
}

template<typename algo_t>
Workspace chooseAlgorithm(
    cudnnHandle_t handle, ConvolutionParams* params,
    bool benchmark,
    const ConvolutionDescriptor& cdesc,
    const TensorDescriptor& idesc, const TensorDescriptor& odesc, const FilterDescriptor& wdesc,
    algo_t* algo)
{
  findAlgorithm(handle, params, benchmark, cdesc, idesc, odesc, wdesc, algo);

  using search = algorithm_search<algo_t>;
  size_t workspace_size;
  search::getWorkspaceSize(handle, cdesc, idesc, odesc, wdesc, *algo, &workspace_size);
  try {
    return Workspace(workspace_size);
  } catch (std::runtime_error& e) {
    cudaGetLastError(); // clear OOM error

    // switch to default algorithm and record it in the cache to prevent
    // further OOM errors
    *algo = search::DEFAULT_ALGO;
    search::cache().insert(*params, *algo);

    search::getWorkspaceSize(handle, cdesc, idesc, odesc, wdesc, *algo, &workspace_size);
    return Workspace(workspace_size);
  }
}

// ---------------------------------------------------------------------
//
// Bias addition
//
// ---------------------------------------------------------------------

// In-place!
void cudnn_convolution_add_bias_(CheckedFrom c, TensorArg output, TensorArg bias)
{
  checkSameType(c, {output, bias});
  checkSameGPU(c, {output, bias});
  checkSize(c, bias, { output->size(output_channels_dim) });

  // See Note [CuDNN broadcast padding].  Handle the left padding
  // ourselves, but use TensorDescriptor's padding argument to do the rest.
  TensorDescriptor bdesc{bias->expand({1, bias->size(0)}), output->dim()};
  TensorDescriptor odesc{*output};

  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(*bias);
  Constant one(dataType, 1);

  CUDNN_CHECK(cudnnAddTensor(handle, &one, bdesc.desc, bdesc.ptr,
                                     &one, odesc.desc, odesc.ptr));
}

// The general strategy: implementation of each convolution function is
// split into three parts:
//    - cudnn_convolution_forward (at::Tensor overload)
//      This is the entry point for clients
//    - cudnn_convolution_forward (TensorArg overload)
//      This is the worker function.  It's factored out from the
//      at::Tensor overload because it is used to implement both
//      convolution forwards, and transposed convolution backwards
//      (TensorArgs is setup so that argument checking gives accurate
//      messages in all cases.)
//    - _cudnn_convolution_forward
//      This is the actual dispatch function, it is a direct wrapper
//      on top of the CuDNN API and, in particular, does not implement
//      legacy group support on old versions of CuDNN.
//
// There's also a 'full' variant that comes with bias addition.
//
// Where does argument checking happen?  Here's the division of
// responsibility:
//  - Things that happen in at::Tensor
//    - TensorArg allocation
//    - cudnnSetStreamToCurrent
//  - Things that happen in TensorArg
//    - Check arguments (type, GPU, shape)
//
// TODO: Consider renaming zero-indexed arguments to "self"



// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

// The raw API directly invokes CuDNN and does not emulate support
// for group convolution on old versions of CuDNN.
//
// There are a few reasons this should never be directly exposed
// via ATen:
//
//    - It takes output as a parameter (this should be computed!)
//    - It doesn't do input checking
//    - It doesn't resize output (it is assumed to be correctly sized)
//    - It takes a ConvolutionParams struct
//
void raw_cudnn_convolution_forward_out(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    ConvolutionParams* params, bool benchmark) {

  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(input);

  ConvolutionDescriptor cdesc;
  cdesc.set(dataType, input.dim() - 2, params->padding, params->stride, params->dilation, params->groups);
  TensorDescriptor idesc{input};
  FilterDescriptor wdesc{weight};
  TensorDescriptor odesc{output};

  // TODO: when we do legacy group convolution support, we'll repeatedly
  // reinitialize the workspace for each convolution we do.  This is
  // wasteful; we'd rather reuse the workspace.  OTOH, legacy group
  // convolution support is already pretty slow, so this might not
  // matter.  (This applies to raw_cudnn_convolution_backward as well.)
  cudnnConvolutionFwdAlgo_t fwdAlg;
  Workspace workspace = chooseAlgorithm(handle, params, benchmark, cdesc, idesc, odesc, wdesc, &fwdAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  CUDNN_CHECK(cudnnConvolutionForward(
    handle, &one, idesc.desc, idesc.ptr,
            wdesc.desc, wdesc.ptr,
            cdesc.desc, fwdAlg, workspace.data, workspace.size,
            &zero, odesc.desc, odesc.ptr));
}

Tensor cudnn_convolution_forward(
    CheckedFrom c,
    TensorArg input, TensorArg weight,
    IntList padding, IntList stride, IntList dilation, int groups,
    bool benchmark, bool deterministic)
{
  checkSameType(c, {input, weight});
  checkSameGPU(c, {input, weight});

  auto output_t = input->type().tensor();
  output_t.resize_(conv_output_size(input->sizes(), weight->sizes(),
                                    padding, stride, dilation, groups));

  // Avoid ambiguity of "output" when this is being used as backwards
  TensorArg output{ output_t, "result", 0 };
  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  ConvolutionParams params;
#if CUDNN_VERSION < 7000
  setConvolutionParams(&params, *input, *weight, padding, stride, dilation, 1, deterministic);
  for (int i = 0; i < groups; i++) {
    raw_cudnn_convolution_forward_out(
        narrowGroup(*output, output_channels_dim, i, groups),
        narrowGroup(*input, input_channels_dim, i, groups),
        narrowGroup(*weight, weight_output_channels_dim, i, groups),
        &params, benchmark);
  }
#else
  setConvolutionParams(&params, *input, *weight, padding, stride, dilation, groups, deterministic);
  raw_cudnn_convolution_forward_out(*output, *input, *weight, &params, benchmark);
#endif

  return *output;
}

Tensor cudnn_convolution_forward(
    const Tensor& input_t, const Tensor& weight_t,
    IntList padding, IntList stride, IntList dilation,
    int groups, bool benchmark, bool deterministic)
{
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 };
  cudnnSetStreamToCurrent();
  return cudnn_convolution_forward(
    "cudnn_convolution_forward",
    input, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor cudnn_convolution_full_forward(
    const Tensor& input_t, const Tensor& weight_t, const Tensor& bias_t,
    IntList padding, IntList stride, IntList dilation,
    int groups, bool benchmark, bool deterministic)
{
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  cudnnSetStreamToCurrent();
  CheckedFrom c = "cudnn_convolution_full_forward";
  auto output_t = cudnn_convolution_forward(
    c, input, weight, padding, stride, dilation, groups, benchmark, deterministic);
  if (bias->defined()) {
    cudnn_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
  }
  return output_t;
}

// NB: output_padding not needed here, as there is no ambiguity to
// resolve
Tensor cudnn_convolution_transpose_backward(
    const Tensor& grad_output_t, const Tensor& weight_t,
    IntList padding, IntList stride, IntList dilation,
    int groups, bool benchmark, bool deterministic)
{
  TensorArg grad_output { grad_output_t,  "grad_output", 1 },
            weight      { weight_t, "weight", 2 };
  cudnnSetStreamToCurrent();
  return cudnn_convolution_forward(
    "cudnn_convolution_transpose_backward",
    grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

// ---------------------------------------------------------------------
//
// Convolution backward / Transposed convolution forward
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_backward_out(
    const at::Tensor& grad_input, const at::Tensor& grad_output, const at::Tensor& weight,
    ConvolutionParams* params, bool benchmark) {

  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(grad_output);

  ConvolutionDescriptor cdesc;
  cdesc.set(dataType, grad_output.dim() - 2, params->padding, params->stride, params->dilation, params->groups);
  TensorDescriptor idesc{grad_input};
  FilterDescriptor wdesc{weight};
  TensorDescriptor odesc{grad_output};

  cudnnConvolutionBwdDataAlgo_t bwdDataAlg;
  Workspace workspace = chooseAlgorithm(handle, params, benchmark, cdesc, idesc, odesc, wdesc, &bwdDataAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  CUDNN_CHECK(cudnnConvolutionBackwardData(
      handle, &one, wdesc.desc, wdesc.ptr,
              odesc.desc, odesc.ptr,
              cdesc.desc, bwdDataAlg, workspace.data, workspace.size,
              &zero, idesc.desc, idesc.ptr));
}

// Backward and transpose are algorithmically equivalent, but they
// compute their geometry differently.  In a backwards, you knew what
// the original size of the input tensor was, so you can cache that
// geometry and fill it directly.  In transposed convolution, it is
// more conventional to not explicitly specify the output (previously
// input) size, and compute it.  This, however, leaves a degree of
// freedom; this degree of freedom is resolved using the
// output_padding parameter.  Both of these interfaces are equivalent,
// but they are differently convenient depending on the use case.

Tensor cudnn_convolution_backward(
    CheckedFrom c,
    IntList input_size, TensorArg grad_output, TensorArg weight,
    IntList padding, IntList stride, IntList dilation, int groups,
    bool benchmark, bool deterministic)
{
  checkSameType(c, {grad_output, weight});
  checkSameGPU(c, {grad_output, weight});

  auto grad_input_t = grad_output->type().tensor();
  grad_input_t.resize_(input_size);

  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{ grad_input_t, "result", 0 };
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  ConvolutionParams params;
#if CUDNN_VERSION < 7000
  setConvolutionParams(&params, *grad_input, *weight, padding, stride, dilation, 1, deterministic);
  for (int i = 0; i < groups; i++) {
    raw_cudnn_convolution_backward_out(
        narrowGroup(*grad_input, input_channels_dim, i, groups),
        narrowGroup(*grad_output, output_channels_dim, i, groups),
        narrowGroup(*weight, weight_output_channels_dim, i, groups),
        &params, benchmark);
  }
#else
  setConvolutionParams(&params, *grad_input, *weight, padding, stride, dilation, groups, deterministic);
  raw_cudnn_convolution_backward_out(*grad_input, *grad_output, *weight, &params, benchmark);
#endif

  return *grad_input;
}

Tensor cudnn_convolution_transpose(
    CheckedFrom c,
    TensorArg grad_output, TensorArg weight,
    IntList padding, IntList output_padding, IntList stride, IntList dilation, int groups,
    bool benchmark, bool deterministic)
{
  auto input_size = conv_input_size(grad_output->sizes(), weight->sizes(),
                                    padding, output_padding, stride, dilation, groups);
  return cudnn_convolution_backward(c, input_size, grad_output, weight,
                                    padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor cudnn_convolution_backward(
    IntList input_size, const Tensor& grad_output_t, const Tensor& weight_t,
    IntList padding, IntList stride, IntList dilation, int groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            weight{ weight_t, "weight", 2 };
  cudnnSetStreamToCurrent();
  return cudnn_convolution_backward(
      "cudnn_convolution_backward",
      input_size, grad_output, weight,
      padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor cudnn_convolution_transpose_full_forward(
    const Tensor& input_t, const Tensor& weight_t, const Tensor& bias_t,
    IntList padding, IntList output_padding, IntList stride, IntList dilation,
    int groups, bool benchmark, bool deterministic)
{
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "cudnn_convolution_transpose_full_forward";
  auto output_t = cudnn_convolution_transpose(
    c, input, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  if (bias->defined()) {
    cudnn_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
  }
  return output_t;
}

// ---------------------------------------------------------------------
//
// Convolution backward (weight)
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_backward_weight_out(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    ConvolutionParams* params, bool benchmark) {

  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(input);

  ConvolutionDescriptor cdesc;
  cdesc.set(dataType, input.dim() - 2, params->padding, params->stride, params->dilation, params->groups);
  TensorDescriptor idesc{input};
  FilterDescriptor wdesc{grad_weight};
  TensorDescriptor odesc{grad_output};

  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlg;
  Workspace workspace = chooseAlgorithm(handle, params, benchmark, cdesc, idesc, odesc, wdesc, &bwdFilterAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  CUDNN_CHECK(cudnnConvolutionBackwardFilter(
      handle, &one, idesc.desc, idesc.ptr,
              odesc.desc, odesc.ptr,
              cdesc.desc, bwdFilterAlg, workspace.data, workspace.size,
              &zero, wdesc.desc, wdesc.ptr));
}

Tensor cudnn_convolution_backward_weight(
    CheckedFrom c,
    IntList weight_size, TensorArg grad_output, TensorArg input,
    IntList padding, IntList stride, IntList dilation, int groups,
    bool benchmark, bool deterministic)
{

  checkSameType(c, {grad_output, input});
  checkSameGPU(c, {grad_output, input});

  auto grad_weight_t = grad_output->type().tensor();
  grad_weight_t.resize_(weight_size);

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_weight{ grad_weight_t, "result", 0 };
  convolution_shape_check(c, input, grad_weight, grad_output, padding, stride, dilation, groups);

  ConvolutionParams params;
#if CUDNN_VERSION < 7000
  setConvolutionParams(&params, *input, *grad_weight, padding, stride, dilation, 1, deterministic);
  for (int i = 0; i < groups; i++) {
    raw_cudnn_convolution_backward_weight_out(
        narrowGroup(*grad_weight, weight_output_channels_dim, i, groups),
        narrowGroup(*grad_output, output_channels_dim, i, groups),
        narrowGroup(*input, input_channels_dim, i, groups),
        &params, benchmark);
  }
#else
  setConvolutionParams(&params, *input, *grad_weight, padding, stride, dilation, groups, deterministic);
  raw_cudnn_convolution_backward_weight_out(*grad_weight, *grad_output, *input, &params, benchmark);
#endif

  return grad_weight_t;
}

Tensor cudnn_convolution_backward_weight(
    IntList weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntList padding, IntList stride, IntList dilation, int groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            input{ input_t, "input", 2 };
  cudnnSetStreamToCurrent();
  return cudnn_convolution_backward_weight(
      "cudnn_convolution_backward_weight",
      weight_size, grad_output, input,
      padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor cudnn_convolution_transpose_backward_weight(
    IntList weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntList padding, IntList stride, IntList dilation, int groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            input{ input_t, "input", 2 };
  cudnnSetStreamToCurrent();
  return cudnn_convolution_backward_weight(
      "cudnn_convolution_backward_weight",
      weight_size, input, grad_output,
      padding, stride, dilation, groups, benchmark, deterministic);
}

// ---------------------------------------------------------------------
//
// Convolution backward (bias)
//
// ---------------------------------------------------------------------

Tensor cudnn_convolution_backward_bias(
    const Tensor& grad_output_t)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 };
  cudnnSetStreamToCurrent();

  auto grad_bias_t = grad_output->type().tensor();
  grad_bias_t.resize_({ grad_output->size(output_channels_dim) });

  TensorArg grad_bias{ grad_bias_t, "result", 0 };

  // See Note [CuDNN broadcast padding].  Handle the left padding
  // ourselves, but use TensorDescriptor's pad argument to do the rest.
  TensorDescriptor bdesc{grad_bias->expand({1, grad_bias->size(0)}), grad_output->dim()};
  TensorDescriptor odesc{*grad_output};

  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(*grad_bias);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  CUDNN_CHECK(cudnnConvolutionBackwardBias(handle, &one, odesc.desc, odesc.ptr,
                                                   &zero, bdesc.desc, bdesc.ptr));
  return *grad_bias;
}

}}  // namespace
