import os
import unittest
import sys

import torch
import torch.utils.cpp_extension
import torch.backends.cudnn
try:
    import torch_test_cpp_extension.cpp as cpp_extension
except ImportError:
    print("\'test_cpp_extensions.py\' cannot be invoked directly. " +
          "Run \'python run_test.py -i cpp_extensions\' for the \'test_cpp_extensions.py\' tests.")
    raise

import common

from torch.utils.cpp_extension import CUDA_HOME
TEST_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
TEST_CUDNN = False
if TEST_CUDA:
    CUDNN_HEADER_EXISTS = os.path.isfile(os.path.join(CUDA_HOME, 'include/cudnn.h'))
    TEST_CUDNN = TEST_CUDA and CUDNN_HEADER_EXISTS and torch.backends.cudnn.is_available()


class TestCppExtension(common.TestCase):


    def test_complex_registration(self):
        cpp_source = '''
        #include <ATen/detail/ComplexHooksInterface.h>
        #include <ATen/detail/VariableHooksInterface.h>
        #include <ATen/Type.h>
        #include <ATen/CPUFloatType.h>

        #include "ATen/TensorImpl.h"
        #include "ATen/CPUGenerator.h"
        #include "ATen/TensorImpl.h"
        #include "ATen/Allocator.h"
        #include "ATen/DeviceGuard.h"
        #include "ATen/NativeFunctions.h"
        #include "ATen/UndefinedTensor.h"
        #include "ATen/Utils.h"
        #include "ATen/WrapDimUtils.h"
        #include "ATen/core/Half.h"
        #include "ATen/core/optional.h"

        #include <cstddef>
        #include <functional>
        #include <memory>
        #include <utility>

        #include "ATen/Config.h"

        namespace at {

        struct CPUComplexFloatType : public at::Type {

          CPUComplexFloatType()
            : Type(CPUTensorId(), /*is_variable=*/false, /*is_undefined=*/false) {}

          virtual ScalarType scalarType() const override;
          virtual Backend backend() const override;
          virtual bool is_cuda() const override;
          virtual bool is_sparse() const override;
          virtual bool is_distributed() const override;
          virtual Storage storage(bool resizable = false) const override;
          virtual Storage storage(size_t size, bool resizable = false) const override;
          virtual Storage storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const override;
          virtual Storage storageWithAllocator(int64_t size, Allocator* allocator) const override;
          virtual std::unique_ptr<Generator> generator() const override;
          virtual const char * toString() const override;
          virtual size_t elementSizeInBytes() const override;
          virtual TypeID ID() const override;
          static const char * typeString();
          virtual Storage unsafeStorageFromTH(void * th_pointer, bool retain) const override;
          virtual Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const override;
          virtual Tensor & s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const override;
          virtual Tensor & _s_copy_from(const Tensor & self, Tensor & dst, bool non_blocking) const override;
        };

        struct ComplexHooks : public at::ComplexHooksInterface {
          ComplexHooks(ComplexHooksArgs) {}
          void registerComplexTypes(Context* context) const override {
            context->registerType(Backend::CPU, ScalarType::ComplexFloat, new CPUComplexFloatType());
          }
        };

        ScalarType CPUComplexFloatType::scalarType() const {
          return ScalarType::ComplexFloat;
        }

        Backend CPUComplexFloatType::backend() const {
          return Backend::CPU;
        }
        bool CPUComplexFloatType::is_cuda() const { return backend() == Backend::CUDA || backend() == Backend::SparseCUDA; }
        bool CPUComplexFloatType::is_sparse() const { return backend() == Backend::SparseCPU || backend() == Backend::SparseCUDA; }
        bool CPUComplexFloatType::is_distributed() const { return false; }

        Storage CPUComplexFloatType::storage(bool resizable) const {
          return Storage(
              ScalarType::ComplexFloat,
              0,
              getTHDefaultAllocator(),
              resizable
          );
        }
        Storage CPUComplexFloatType::storage(size_t size, bool resizable) const {
          return Storage(
              ScalarType::ComplexFloat,
              size,
              getTHDefaultAllocator(),
              resizable
          );
        }
        Storage CPUComplexFloatType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
            return Storage(
              ScalarType::ComplexFloat,
              InefficientStdFunctionContext::makeDataPtr(data, deleter,
                DeviceType::CPU
              ),
              size,
              deleter);
        }
        Storage CPUComplexFloatType::storageWithAllocator(int64_t size, Allocator* allocator) const {
            return Storage(ScalarType::ComplexFloat, size, allocator);
        }
        Tensor CPUComplexFloatType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
          return Tensor(static_cast<TensorImpl*>(th_pointer), retain);
        }
        Storage CPUComplexFloatType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
          if (retain)
            THFloatStorage_retain( (THFloatStorage*) th_pointer);
          return Storage((THFloatStorage*) th_pointer);
        }
        std::unique_ptr<Generator> CPUComplexFloatType::generator() const {
          // Hmm, this is weird.
          return nullptr;
          // return std::unique_ptr<Generator>(new CPUGenerator(&at::globalContext()));
        }

        const char * CPUComplexFloatType::toString() const {
          return CPUComplexFloatType::typeString();
        }
        TypeID CPUComplexFloatType::ID() const {
          return TypeID::CPUComplexFloat;
        }

        size_t CPUComplexFloatType::elementSizeInBytes() const {
          return sizeof(float);
        }

        const char * CPUComplexFloatType::typeString() {
          return "CPUComplexFloatType";
        }

        Tensor & CPUComplexFloatType::s_copy_(Tensor & dst, const Tensor & src, bool non_blocking) const {
          AT_ERROR("not yet supported");
        }

        Tensor & CPUComplexFloatType::_s_copy_from(const Tensor & src, Tensor & dst, bool non_blocking) const {
          AT_ERROR("not yet supported");
        }

        REGISTER_COMPLEX_HOOKS(ComplexHooks);

        } // namespace at
        '''

        module = torch.utils.cpp_extension.load_inline(
            name='complex_registration_extension',
            cpp_sources=cpp_source,
            functions=[],
            verbose=True)

        torch.zeros(0, dtype=torch.complex64)


if __name__ == '__main__':
    common.run_tests()
