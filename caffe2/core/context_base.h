#pragma once

#include <cstdlib>
#include <ctime>
#include <random>
#include <unordered_map>

#include "caffe2/core/allocator.h"
#include "caffe2/core/event.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/typeid.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

/**
 * Virtual interface for the Context class in Caffe2.
 *
 * A Context defines all the necessities to run an operator on a specific
 * device. Specific Context classes have the freedom to choose what functions it
 * implements, but there are a few functions that you should consider
 * implementing if you want to write your own context class:
 * - void SwitchToDevice(): any necessary code to switch to the device before
 *     running anything.
 * - void WaitEvent(const Event& ev): make the current context to wait on
 *     an event. For example, for cuda, this is the equivalent of
 *     cudaStreamWaitEvent. For CPU context, it essentially synchronizes the
 *     event.
 * - void Record(Event* ev): record the async activities on the current context
 *     to the event. For example, for cuda, this is the equivalent of
 *     cudaEventRecord on the current stream. For CPU context, it is always
 *     synchronous.
 * - void FinishDeviceComputation(): any wrapping-up work after all the
 *     computation of the operator is done. If there are errors during the
 *     execution, throw exception. For example, in a CUDAContext, this function
 *     carries out a stream synchronization and spots potential errors for
 *     the cuda kernel calls.
 * - static std::pair<void*, MemoryDeleter> New(size_t nbytes): allocates
       memory and returns a deleter.
 * - template <class SrcContext, class DstContext> void CopyBytes(...): does
 *     cross context memory copy.
 * - template <typename T, class SrcContext, class DstContext> void Copy(...):
 *     usually a simple wrapper around the above CopyBytes function.
 */
class BaseContext {
 public:
  virtual ~BaseContext() noexcept {}

  virtual void SwitchToDevice(int /*stream_id*/) = 0;

  inline void SwitchToDevice() {
    SwitchToDevice(0);
  }

  virtual void WaitEvent(const Event& ev) = 0;

  virtual void Record(Event* ev, const char* err_msg = nullptr) const = 0;

  virtual void FinishDeviceComputation() = 0;

  // This used to be arbitrary cross-device copy, but it turns out everyone
  // did direct CPU-X copy, so we just make three functions for it (to avoid
  // double dispatch).  This will get obsoleted by C10.

  virtual void CopyBytesSameDevice(size_t nbytes, const void* src, void* dst) = 0;

  virtual void CopyBytesFromCPU(size_t nbytes, const void* src, void* dst) = 0;

  virtual void CopyBytesToCPU(size_t nbytes, const void* src, void* dst) = 0;

  template <typename T>
  inline void CopySameDevice(size_t n, const T* src, T* dst) {
    if (std::is_fundamental<T>::value) {
      CopyBytesSameDevice(
          n * sizeof(T),
          static_cast<const void*>(src),
          static_cast<void*>(dst));
    } else {
      for (int i = 0; i < n; ++i) {
        dst[i] = src[i];
      }
    }
  }

  template <typename T>
  inline void CopyFromCPU(size_t n, const T* src, T* dst) {
    if (std::is_fundamental<T>::value) {
      CopyFromCPU(
          n * sizeof(T),
          static_cast<const void*>(src),
          static_cast<void*>(dst));
    } else {
      // Hmmmm.... this probably won't work for non-CPU contexts
      for (int i = 0; i < n; ++i) {
        dst[i] = src[i];
      }
    }
  }

  template <typename T>
  inline void CopyToCPU(size_t n, const T* src, T* dst) {
    if (std::is_fundamental<T>::value) {
      CopyToCPU(
          n * sizeof(T),
          static_cast<const void*>(src),
          static_cast<void*>(dst));
    } else {
      // Hmmmm.... this probably won't work for non-CPU contexts
      for (int i = 0; i < n; ++i) {
        dst[i] = src[i];
      }
    }
  }

  virtual void EnforceMetaCopyOK() {
    // TODO: say what type of context
    CAFFE_ENFORCE(0, "Context requires fundamental types");
  }

  inline void
  CopyItemsSameDevice(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    if (meta.copy()) {
      EnforceMetaCopyOK();
      meta.copy()(src, dst, n);
    } else {
      CopyBytesSameDevice(n * meta.itemsize(), src, dst);
    }
  }

  inline void
  CopyItemsFromCPU(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    if (meta.copy()) {
      EnforceMetaCopyOK();
      meta.copy()(src, dst, n);
    } else {
      CopyBytesFromCPU(n * meta.itemsize(), src, dst);
    }
  }

  inline void
  CopyItemsToCPU(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    if (meta.copy()) {
      EnforceMetaCopyOK();
      meta.copy()(src, dst, n);
    } else {
      CopyBytesToCPU(n * meta.itemsize(), src, dst);
    }
  }


#if 0
  virtual std::pair<void*, MemoryDeleter> New(size_t nbytes) = 0;

  virtual bool HasAsyncPartDefault() = 0;

  virtual bool SupportsAsyncScheduling() = 0;

  virtual bool IsStreamFree(const DeviceOption& /* unused */, int /* unused */) = 0;
#endif
};

} // namespace caffe2
