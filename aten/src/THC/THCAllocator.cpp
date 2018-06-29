#include "THCAllocator.h"

struct THCudaHostDeleter : public at::Deleter {
  void deallocate(void* ctx, void* ptr) const override {
    if (!ptr) return;

    THCudaCheck(cudaFreeHost(ptr));
  }
  static at::BoundDeleter make() {
    return {&singleton_, nullptr};
  }
private:
  static THCudaHostDeleter singleton_;
};
THCudaHostDeleter THCudaHostDeleter::singleton_;

struct THCudaHostAllocator : public at::Allocator {
  std::unique_ptr<void, at::BoundDeleter> allocate(size_t size) const override {
    void* ptr;

    if (size < 0) THError("Invalid memory size: %ld", size);

    if (size == 0) return NULL;

    THCudaCheck(cudaMallocHost(&ptr, size));

    return {ptr, THCudaHostDeleter::make()};
  }
};

static THCudaHostAllocator th_cuda_host_allocator;
at::Allocator* getTHCudaHostAllocator() {
  return &th_cuda_host_allocator;
}

struct THCIpcDeleter : public at::Deleter {
  void deallocate(void* ctx, void* ptr) const override {
    int prev_device;
    int device = (int)(int64_t)ctx;

    THCudaCheck(cudaGetDevice(&prev_device));
    THCudaCheck(cudaSetDevice(device));
    THCudaCheck(cudaIpcCloseMemHandle(ptr));
    THCudaCheck(cudaSetDevice(prev_device));
  }

  static at::BoundDeleter make(int device) {
    // TODO: Do this properly with intptr_t (but is it portable enough?)
    return {&singleton_, (void*)(int64_t)device};
  }
private:
  static THCIpcDeleter singleton_;
};
THCIpcDeleter THCIpcDeleter::singleton_;

at::BoundDeleter makeTHCIpcDeleter(int device) {
  return THCIpcDeleter::make(device);
}

struct THCIpcAllocator : public at::Allocator {
  std::unique_ptr<void, at::BoundDeleter> allocate(size_t size) const override {
    AT_ERROR("THCIpcAllocator::allocate() not supported");
  }
};

static THCIpcAllocator thc_ipc_allocator;
at::Allocator* getTHCIpcAllocator() {
  return &thc_ipc_allocator;
}

struct THCUVADeleter : public at::Deleter {
  void deallocate(void* ctx, void* ptr) const override {
    if (!ptr) return;
    THCudaCheck(cudaFree(ptr));
  }
  static at::BoundDeleter make() {
    return {&singleton_, nullptr};
  }
private:
  static THCUVADeleter singleton_;
};
THCUVADeleter THCUVADeleter::singleton_;

struct THCUVAAllocator : public at::Allocator {
  std::unique_ptr<void, at::BoundDeleter> allocate(size_t size) const override {
    AT_CHECK(size >= 0, "Invalid memory size: ", size);

    if (size == 0) return NULL;

    // See J.1.1 of the CUDA_C_Programming_Guide.pdf for UVA and coherence rules
    // on various compute capabilities.
    void* ptr;
    THCudaCheck(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
    return {ptr, THCUVADeleter::make()};
  }
};

static THCUVAAllocator thc_uva_allocator;
at::Allocator* getTHCUVAAllocator() {
  return &thc_uva_allocator;
}
