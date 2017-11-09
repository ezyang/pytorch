#pragma once

#include "cudnn-wrapper.h"

namespace at { namespace cudnn {

cudnnHandle_t getCudnnHandle();

}} // namespace
