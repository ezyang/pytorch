#pragma once

#include <c10/core/SymInt.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <array>
#include <initializer_list>
#include <iterator>
#include <vector>

namespace c10 {
//using SymIntArrayRef = ArrayRef<SymInt>;

class SymIntArrayRef final {
  explicit SymIntArrayRef() {}
};

TORCH_API at::IntArrayRef asIntArrayRefSlow(c10::SymIntArrayRef ar);
TORCH_API at::IntArrayRef asIntArrayRefUnchecked(c10::SymIntArrayRef ar);
TORCH_API c10::optional<at::IntArrayRef> asIntArrayRefSlowOpt(
    c10::SymIntArrayRef ar);

// Prefer using a more semantic constructor, like
// fromIntArrayRefKnownNonNegative
inline SymIntArrayRef fromIntArrayRefUnchecked(IntArrayRef array_ref) {
  return SymIntArrayRef(
      reinterpret_cast<const SymInt*>(array_ref.data()), array_ref.size());
}

inline SymIntArrayRef fromIntArrayRefKnownNonNegative(IntArrayRef array_ref) {
  return fromIntArrayRefUnchecked(array_ref);
}

inline SymIntArrayRef fromIntArrayRef(IntArrayRef array_ref) {
  for (size_t i = 0; i < array_ref.size(); ++i) {
    TORCH_CHECK(
        SymInt::check_range(array_ref[i]),
        "IntArrayRef contains an int that cannot be represented as a SymInt: ",
        array_ref[i]);
  }
  return SymIntArrayRef(
      reinterpret_cast<const SymInt*>(array_ref.data()), array_ref.size());
}


template <>
class ArrayRef<SymInt> final {
  using T = SymInt;
 public:
  using iterator = const T*;
  using const_iterator = const T*;
  using size_type = size_t;
  using value_type = T;

  using reverse_iterator = std::reverse_iterator<iterator>;

 private:
  /// The start of the array, in an external buffer.
  const T* Data;

  /// The number of elements.
  size_type Length;

  void debugCheckNullptrInvariant() {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        Data != nullptr || Length == 0,
        "created ArrayRef with nullptr and non-zero length! c10::optional relies on this being illegal");
  }

 public:
  /// @name Constructors
  /// @{

  /// Construct an empty ArrayRef.
  explicit ArrayRef() : Data(nullptr), Length(0) {}

  explicit ArrayRef(const ArrayRef&) = default;
  explicit ArrayRef(ArrayRef&&) = default;

  /// Construct an ArrayRef from a single element.
  explicit ArrayRef(const T& OneElt) : Data(&OneElt), Length(1) {}

  /// Construct an ArrayRef from a pointer and length.
  explicit ArrayRef(const T* data, size_t length)
      : Data(data), Length(length) {
    debugCheckNullptrInvariant();
  }

  /// Construct an ArrayRef from a range.
  explicit ArrayRef(const T* begin, const T* end)
      : Data(begin), Length(end - begin) {
    debugCheckNullptrInvariant();
  }

#if 0
  /// Construct an ArrayRef from a SmallVector. This is templated in order to
  /// avoid instantiating SmallVectorTemplateCommon<T> whenever we
  /// copy-construct an ArrayRef.
  template <typename U>
  explicit ArrayRef(const SmallVectorTemplateCommon<T, U>& Vec)
      : Data(Vec.data()), Length(Vec.size()) {
    debugCheckNullptrInvariant();
  }

  template <
      typename Container,
      typename = std::enable_if_t<std::is_same<
          std::remove_const_t<decltype(std::declval<Container>().data())>,
          T*>::value>>
  explicit ArrayRef(const Container& container)
      : Data(container.data()), Length(container.size()) {
    debugCheckNullptrInvariant();
  }

  /// Construct an ArrayRef from a std::vector.
  // The enable_if stuff here makes sure that this isn't used for
  // std::vector<bool>, because ArrayRef can't work on a std::vector<bool>
  // bitfield.
  template <typename A>
  explicit ArrayRef(const std::vector<T, A>& Vec)
      : Data(Vec.data()), Length(Vec.size()) {
    static_assert(
        !std::is_same<T, bool>::value,
        "ArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.");
  }

  /// Construct an ArrayRef from a std::array
  template <size_t N>
  explicit ArrayRef(const std::array<T, N>& Arr)
      : Data(Arr.data()), Length(N) {}

  /// Construct an ArrayRef from a C array.
  template <size_t N>
  explicit ArrayRef(const T (&Arr)[N]) : Data(Arr), Length(N) {}
#endif

  /// @}
  /// @name Simple Operations
  /// @{

  iterator begin() const {
    return Data;
  }
  iterator end() const {
    return Data + Length;
  }

  // These are actually the same as iterator, since ArrayRef only
  // gives you const iterators.
  const_iterator cbegin() const {
    return Data;
  }
  const_iterator cend() const {
    return Data + Length;
  }

  reverse_iterator rbegin() const {
    return reverse_iterator(end());
  }
  reverse_iterator rend() const {
    return reverse_iterator(begin());
  }

  /// empty - Check if the array is empty.
  bool empty() const {
    return Length == 0;
  }

  const T* data() const {
    return Data;
  }

  /// size - Get the array size.
  size_t size() const {
    return Length;
  }

  /// front - Get the first element.
  const T& front() const {
    TORCH_CHECK(
        !empty(), "ArrayRef: attempted to access front() of empty list");
    return Data[0];
  }

  /// back - Get the last element.
  const T& back() const {
    TORCH_CHECK(!empty(), "ArrayRef: attempted to access back() of empty list");
    return Data[Length - 1];
  }

  /// equals - Check for element-wise equality.
  bool equals(ArrayRef RHS) const {
    return Length == RHS.Length && std::equal(begin(), end(), RHS.begin());
  }

  /// slice(n, m) - Take M elements of the array starting at element N
  ArrayRef<T> slice(size_t N, size_t M)
      const {
    TORCH_CHECK(
        N + M <= size(),
        "ArrayRef: invalid slice, N = ",
        N,
        "; M = ",
        M,
        "; size = ",
        size());
    return ArrayRef<T>(data() + N, M);
  }

  /// slice(n) - Chop off the first N elements of the array.
  ArrayRef<T> slice(size_t N) const {
    return slice(N, size() - N);
  }

  /// @}
  /// @name Operator Overloads
  /// @{
  const T& operator[](size_t Index) const {
    return Data[Index];
  }

  /// Vector compatibility
  const T& at(size_t Index) const {
    TORCH_CHECK(
        Index < Length,
        "ArrayRef: invalid index Index = ",
        Index,
        "; Length = ",
        Length);
    return Data[Index];
  }

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type&
  operator=(U&& Temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type&
  operator=(std::initializer_list<U>) = delete;

  /// @}
  /// @name Expensive Operations
  /// @{
  std::vector<T> vec() const {
    return std::vector<T>(Data, Data + Length);
  }

  /// @}
};

TORCH_API at::IntArrayRef asIntArrayRefSlow(c10::SymIntArrayRef ar);
TORCH_API at::IntArrayRef asIntArrayRefUnchecked(c10::SymIntArrayRef ar);
TORCH_API c10::optional<at::IntArrayRef> asIntArrayRefSlowOpt(
    c10::SymIntArrayRef ar);

// Prefer using a more semantic constructor, like
// fromIntArrayRefKnownNonNegative
inline SymIntArrayRef fromIntArrayRefUnchecked(IntArrayRef array_ref) {
  return SymIntArrayRef(
      reinterpret_cast<const SymInt*>(array_ref.data()), array_ref.size());
}

inline SymIntArrayRef fromIntArrayRefKnownNonNegative(IntArrayRef array_ref) {
  return fromIntArrayRefUnchecked(array_ref);
}

inline SymIntArrayRef fromIntArrayRef(IntArrayRef array_ref) {
  for (size_t i = 0; i < array_ref.size(); ++i) {
    TORCH_CHECK(
        SymInt::check_range(array_ref[i]),
        "IntArrayRef contains an int that cannot be represented as a SymInt: ",
        array_ref[i]);
  }
  return SymIntArrayRef(
      reinterpret_cast<const SymInt*>(array_ref.data()), array_ref.size());
}

} // namespace c10
