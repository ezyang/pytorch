#pragma once

#include <c10/macros/Macros.h>

template <typename T>
C10_HOST_DEVICE T add(T self, T other, T alpha) {
  return self + alpha * other;
}
