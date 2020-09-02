#include <iostream>
#include <torch/script.h>
#include "common.h"

#include <valgrind/callgrind.h>

void add_s1_grad(int count) {
  auto a = torch::ones({1}, at::TensorOptions().requires_grad(true));
  auto b = torch::ones({1}, at::TensorOptions().requires_grad(true));
  auto c = a;

  benchmark("add_s1_grad", count, [&]() {
    c = at::add(c, b);
  });
}

void add_s1_nograd(int count) {
  torch::NoGradGuard guard;
  auto a = torch::ones({1});
  auto b = torch::ones({1});
  auto c = a;

  benchmark("add_s1_nograd", count, [&]() {
    c = at::add(c, b);
  });
}

void add_s1_nograd_outplace(int count) {
  torch::NoGradGuard nograd_guard;
  auto a = torch::ones({1});
  auto b = torch::ones({1});
  auto c = torch::empty({1});

  benchmark("add_s1_nograd_outplace", count, [&]() {
    at::add_out(c, a, b);
    std::swap(a, c);
  });
}

void add_s1_nograd_outplace_novar(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({1});
  auto b = torch::ones({1});
  auto c = torch::empty({1});

  benchmark("add_s1_nograd_outplace_novar", count, [&]() {
    at::add_out(c, a, b);
    std::swap(a, c);
  });
}

void mm_sN_grad(int count, int N) {
  auto a = torch::ones({N, N}, at::TensorOptions().requires_grad(true));
  auto b = torch::ones({N, N}, at::TensorOptions().requires_grad(true));

  benchmark("mm_s" + std::to_string(N) + "_grad", count, [&]() {
    auto c = at::mm(a, b);
  });
}

void mm_sN_nograd_novar(int count, int N) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({N, N}, at::TensorOptions().requires_grad(false));
  auto b = torch::ones({N, N}, at::TensorOptions().requires_grad(false));

  benchmark("mm_s" + std::to_string(N) + "_nograd_novar", count, [&]() {
    auto c = at::mm(a, b);
  });
}

void mm_sN_nograd_novar_outplace(int count, int N) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({N, N}, at::TensorOptions().requires_grad(false));
  auto b = torch::ones({N, N}, at::TensorOptions().requires_grad(false));
  auto c = torch::empty({N, N}, at::TensorOptions().requires_grad(false));

  benchmark("mm_s" + std::to_string(N) + "_nograd_novar_outplace", count, [&]() {
    at::mm_out(c, a, b);
  });
}

int main() {
  //report_header();

  CALLGRIND_START_INSTRUMENTATION;
  add_s1_nograd_outplace_novar(10000);
  CALLGRIND_STOP_INSTRUMENTATION;
  //add_s1_nograd_outplace(1000000);
  //add_s1_nograd(1000000);
  //add_s1_grad(1000000);

  /*
  for (int N : {64, 256}) {
    std::cout << std::endl;
    int count = (1024 / N) * (1024 / N) * (1024 / N) * 10;
    mm_sN_nograd_novar_outplace(count, N);
    mm_sN_nograd_novar(count, N);
    mm_sN_grad(count, N);
  }
  */

  return 0;
}
