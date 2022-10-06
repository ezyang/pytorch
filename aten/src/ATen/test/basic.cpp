#include <ATen/ops/empty.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/empty_ops.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/cuda.h>
#include <torch/library.h>
#include <ATen/test/test_assert.h>
#include <c10/util/irange.h>
#include <c10/util/CallOnce.h>

#include <iostream>
#include <chrono>
// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <string.h>
#include <sstream>
#include <thread>
#include <mutex>

#include <benchmark/benchmark.h>

using namespace c10;
using namespace at;

static void BM_Empty(benchmark::State& state) {
  std::vector<int64_t> shape = {0,1,2,3,4};
  for (auto _ : state) {
    at::empty(shape);
  }
}
BENCHMARK(BM_Empty);

static void BM_EmptySymInt(benchmark::State& state) {
  std::vector<c10::SymInt> shape = {0,1,2,3,4};
  for (auto _ : state) {
    at::empty_symint(shape);
  }
}
BENCHMARK(BM_EmptySymInt);

static void BM_NativeEmpty(benchmark::State& state) {
  std::vector<int64_t> shape = {0,1,2,3,4};
  for (auto _ : state) {
    at::native::empty_cpu(shape);
  }
}
BENCHMARK(BM_NativeEmpty);

#if 0
at::Tensor empty_memory_format(c10::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  DispatchKeySet _dk = c10::DispatchKeySet(c10::computeDispatchKey(dtype, layout, device));
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow(at::_ops::empty_memory_format::name, at::_ops::empty_memory_format::overload_name)
      .typed<at::Tensor(c10::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format)>();
  return op.redispatch(
      _dk, size, dtype, layout, device, pin_memory, memory_format);
}

TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
  m.impl("aten::empty.memory_format", TORCH_FN(empty_memory_format));
}
#endif

/*
at::Tensor wrapper_memory_format_empty(c10::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  return at::native::empty_cpu(size, dtype, layout, device, pin_memory, memory_format);
}
*/

at::Tensor wrapper_memory_format_empty(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    // No device check
  // DeviceGuard omitted
  for (const auto& s : size) {
    if (s.is_symbolic()) {
      TORCH_CHECK(0, "bad size");
    }
  }
  for (const auto& s : size) {
    if (s.is_symbolic()) {
      TORCH_CHECK(0, "bad size");
    }
  }
  for (const auto& s : size) {
    if (s.is_symbolic()) {
      TORCH_CHECK(0, "bad size");
    }
  }
  for (const auto& s : size) {
    if (s.is_symbolic()) {
      TORCH_CHECK(0, "bad size");
    }
  }
  for (const auto& s : size) {
    if (s.is_symbolic()) {
      TORCH_CHECK(0, "bad size");
    }
  }
  for (const auto& s : size) {
    if (s.is_symbolic()) {
      TORCH_CHECK(0, "bad size");
    }
  }
  for (const auto& s : size) {
    if (s.is_symbolic()) {
      TORCH_CHECK(0, "bad size");
    }
  }
  for (const auto& s : size) {
    if (s.is_symbolic()) {
      TORCH_CHECK(0, "bad size");
    }
  }
  return at::native::empty_cpu(c10::asIntArrayRefUnchecked(size), dtype, layout, device, pin_memory, memory_format);
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("aten::empty.memory_format", TORCH_FN(wrapper_memory_format_empty));
}

#if 0
static void BM_OpsEmptyNoSymInt(benchmark::State& state) {
  c10::InferenceMode mode;
  std::vector<int64_t> shape = {0,1,2,3,4};
  auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow(at::_ops::empty_memory_format::name, at::_ops::empty_memory_format::overload_name)
      .typed<at::Tensor(c10::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format)>();
  for (auto _ : state) {
    op.call(shape, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
  }
}
BENCHMARK(BM_OpsEmptyNoSymInt);
#endif

BENCHMARK_MAIN();
