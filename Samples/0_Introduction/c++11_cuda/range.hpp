/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef UTIL_LANG_RANGE_HPP
#define UTIL_LANG_RANGE_HPP

#include <iterator>
#include <type_traits>

// Make these ranges usable inside CUDA C++ device code
#ifdef __CUDACC__
#define DEVICE_CALLABLE __host__ __device__
#else
#define DEVICE_CALLABLE
#endif

namespace util {
namespace lang {

namespace detail {

template <typename T>
struct range_iter_base : std::iterator<std::input_iterator_tag, T> {
  DEVICE_CALLABLE
  range_iter_base(T current) : current(current) {}

  DEVICE_CALLABLE
  T operator*() const { return current; }

  DEVICE_CALLABLE
  T const* operator->() const { return &current; }

  DEVICE_CALLABLE
  range_iter_base& operator++() {
    ++current;
    return *this;
  }

  DEVICE_CALLABLE
  range_iter_base operator++(int) {
    auto copy = *this;
    ++*this;
    return copy;
  }

  DEVICE_CALLABLE
  bool operator==(range_iter_base const& other) const {
    return current == other.current;
  }

  DEVICE_CALLABLE
  bool operator!=(range_iter_base const& other) const {
    return not(*this == other);
  }

 protected:
  T current;
};

}  // namespace detail

template <typename T>
struct range_proxy {
  struct iter : detail::range_iter_base<T> {
    DEVICE_CALLABLE
    iter(T current) : detail::range_iter_base<T>(current) {}
  };

  struct step_range_proxy {
    struct iter : detail::range_iter_base<T> {
      DEVICE_CALLABLE
      iter(T current, T step)
          : detail::range_iter_base<T>(current), step(step) {}

      using detail::range_iter_base<T>::current;

      DEVICE_CALLABLE
      iter& operator++() {
        current += step;
        return *this;
      }

      DEVICE_CALLABLE
      iter operator++(int) {
        auto copy = *this;
        ++*this;
        return copy;
      }

      // Loses commutativity. Iterator-based ranges are simply broken. :-(
      DEVICE_CALLABLE
      bool operator==(iter const& other) const {
        return step > 0 ? current >= other.current : current < other.current;
      }

      DEVICE_CALLABLE
      bool operator!=(iter const& other) const { return !(*this == other); }

     private:
      T step;
    };

    DEVICE_CALLABLE
    step_range_proxy(T begin, T end, T step)
        : begin_(begin, step), end_(end, step) {}

    DEVICE_CALLABLE
    iter begin() const { return begin_; }

    DEVICE_CALLABLE
    iter end() const { return end_; }

   private:
    iter begin_;
    iter end_;
  };

  DEVICE_CALLABLE
  range_proxy(T begin, T end) : begin_(begin), end_(end) {}

  DEVICE_CALLABLE
  step_range_proxy step(T step) { return {*begin_, *end_, step}; }

  DEVICE_CALLABLE
  iter begin() const { return begin_; }

  DEVICE_CALLABLE
  iter end() const { return end_; }

 private:
  iter begin_;
  iter end_;
};

template <typename T>
struct infinite_range_proxy {
  struct iter : detail::range_iter_base<T> {
    DEVICE_CALLABLE
    iter(T current = T()) : detail::range_iter_base<T>(current) {}

    DEVICE_CALLABLE
    bool operator==(iter const&) const { return false; }

    DEVICE_CALLABLE
    bool operator!=(iter const&) const { return true; }
  };

  struct step_range_proxy {
    struct iter : detail::range_iter_base<T> {
      DEVICE_CALLABLE
      iter(T current = T(), T step = T())
          : detail::range_iter_base<T>(current), step(step) {}

      using detail::range_iter_base<T>::current;

      DEVICE_CALLABLE
      iter& operator++() {
        current += step;
        return *this;
      }

      DEVICE_CALLABLE
      iter operator++(int) {
        auto copy = *this;
        ++*this;
        return copy;
      }

      DEVICE_CALLABLE
      bool operator==(iter const&) const { return false; }

      DEVICE_CALLABLE
      bool operator!=(iter const&) const { return true; }

     private:
      T step;
    };

    DEVICE_CALLABLE
    step_range_proxy(T begin, T step) : begin_(begin, step) {}

    DEVICE_CALLABLE
    iter begin() const { return begin_; }

    DEVICE_CALLABLE
    iter end() const { return iter(); }

   private:
    iter begin_;
  };

  DEVICE_CALLABLE
  infinite_range_proxy(T begin) : begin_(begin) {}

  DEVICE_CALLABLE
  step_range_proxy step(T step) { return step_range_proxy(*begin_, step); }

  DEVICE_CALLABLE
  iter begin() const { return begin_; }

  DEVICE_CALLABLE
  iter end() const { return iter(); }

 private:
  iter begin_;
};

template <typename T>
DEVICE_CALLABLE range_proxy<T> range(T begin, T end) {
  return {begin, end};
}

template <typename T>
DEVICE_CALLABLE infinite_range_proxy<T> range(T begin) {
  return {begin};
}

namespace traits {

template <typename C>
struct has_size {
  template <typename T>
  static constexpr auto check(T*) ->
      typename std::is_integral<decltype(std::declval<T const>().size())>::type;

  template <typename>
  static constexpr auto check(...) -> std::false_type;

  using type = decltype(check<C>(0));
  static constexpr bool value = type::value;
};

}  // namespace traits

template <typename C,
          typename = typename std::enable_if<traits::has_size<C>::value>>
DEVICE_CALLABLE auto indices(C const& cont)
    -> range_proxy<decltype(cont.size())> {
  return {0, cont.size()};
}

template <typename T, std::size_t N>
DEVICE_CALLABLE range_proxy<std::size_t> indices(T(&)[N]) {
  return {0, N};
}

template <typename T>
range_proxy<typename std::initializer_list<T>::size_type> DEVICE_CALLABLE
indices(std::initializer_list<T>&& cont) {
  return {0, cont.size()};
}
}
}  // namespace util::lang

#endif  // ndef UTIL_LANG_RANGE_HPP