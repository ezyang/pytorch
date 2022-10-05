#pragma once

#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/impl/SizesAndStrides.h>
#include <c10/util/SmallVector.h>
#include <cstdint>

namespace c10 {

constexpr size_t kDimVectorStaticSize = C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE;

/// A container for sizes or strides
using DimVector = SmallVector<int64_t, kDimVectorStaticSize>;
using SymDimVector = SmallVector<c10::SymInt, kDimVectorStaticSize>;

// This SymDimVector can be efficiently converted into a SymIntArrayRef
// as it knows if its contents actually contain a symbolic integer.
// For this to work, this needs to have a substantially minimized interface
// that doesn't allow mutating operations
class SymDimVectorWithIsSymbolic {
  SymDimVector data_;
  bool is_symbolic_;

  bool computeSymbolic() const {
    for (const auto& s : data_) {
      if (s.is_symbolic()) return true;
    }
    return false;
  }

  void debugCheckIsSymbolicInvariant() {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        is_symbolic_ == computeSymbolic(),
        "created SymDimVectorWithIsSymbolic with incorrect IsSymbolic tag; real tag is ",
        !is_symbolic_);
  }

public:
  enum Slow { SLOW };
  enum KnownNonSymbolic { KNOWN_NON_SYMBOLIC };
  enum KnownSymbolic { KNOWN_SYMBOLIC };

  using element_type = c10::SymInt;

  SymDimVectorWithIsSymbolic() : data_(), is_symbolic_(false) {}
  SymDimVectorWithIsSymbolic(SymIntArrayRef sa) : data_(sa), is_symbolic_(sa.is_symbolic()) {}
  // We don't mark this as SLOW, because moving it in means we freshly
  // created the vector, which means we've already paid O(n) and it's OK
  // to do another O(n) scan
  SymDimVectorWithIsSymbolic(SymDimVector&& data) : data_(std::move(data)), is_symbolic_(computeSymbolic()) {}
  SymDimVectorWithIsSymbolic(KnownNonSymbolic, SymDimVector&& data) : data_(std::move(data)), is_symbolic_(false) {
    debugCheckIsSymbolicInvariant();
  }
  SymDimVectorWithIsSymbolic(KnownSymbolic, SymDimVector&& data) : data_(std::move(data)), is_symbolic_(true) {
    debugCheckIsSymbolicInvariant();
  }
  SymDimVectorWithIsSymbolic(std::initializer_list<c10::SymInt> data) : data_(data), is_symbolic_(computeSymbolic()) {}

  bool is_symbolic() const {
    return is_symbolic_;
  }

  operator SymIntArrayRef() const {
    return SymIntArrayRef(SymIntArrayRef::UNCHECKED, data_.data(), data_.size(), true);
  }

  size_t size() const { return data_.size(); }
  const SymInt* begin() const { return data_.begin(); }
  const SymInt* end() const { return data_.end(); }

  const SymInt& at(size_t idx) const { return data_.at(idx); }
  const SymInt& operator[](size_t idx) const { return data_[idx]; }

  // NB: this also clones the constituent SymInts
  void cloneFrom(SymIntArrayRef src) {
    data_.clear();
    data_.reserve(src.size());
    for (const auto& s : src) {
      data_.emplace_back(s.clone());
    }
    is_symbolic_ = src.is_symbolic();
  };

  // NB: bounds are unchecked
  void transpose(int64_t dim0, int64_t dim1) {
    std::swap(data_[dim0], data_[dim1]);
    // NB: does not affect is_symbolic status
  }
};

} // namespace c10
