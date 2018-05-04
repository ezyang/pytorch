#include <ATen/detail/CUDAHooksInterface.h>

#include <ATen/Generator.h>

namespace at { namespace cuda { namespace detail {

// The real implementation of CUDAHooksInterface
struct CUDAHooks : public at::detail::CUDAHooksInterface {
  std::unique_ptr<THCState, void(*)(THCState*)> initCUDA() const override;
  std::unique_ptr<Generator> initCUDAGenerator(Context*) const override;
  bool hasCUDA() const override;
  cudaStream_t getCurrentCUDAStream(THCState*) const override;
  struct cudaDeviceProp* getCurrentDeviceProperties(THCState*) const override;
  struct cudaDeviceProp* getDeviceProperties(THCState*, int device) const override;
  int64_t current_device() const override;
  std::unique_ptr<Allocator> newPinnedMemoryAllocator() const override;
  void registerCUDATypes(Context*) const override;
};

}}} // at::cuda::detail

// Unfortunately, REGISTER_CUDA_HOOKS doesn't understand how to absolutely
// qualify reference to at::detail::CUDAHooksRegistry
namespace at { namespace detail {
REGISTER_CUDA_HOOKS(at::cuda::detail::CUDAHooks);
}};
