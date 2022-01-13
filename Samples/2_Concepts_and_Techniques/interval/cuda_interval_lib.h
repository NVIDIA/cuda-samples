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

#ifndef CUDA_INTERVAL_LIB_H
#define CUDA_INTERVAL_LIB_H

#include "cuda_interval_rounded_arith.h"

// Interval template class and basic operations
// Interface inspired from the Boost Interval library (www.boost.org)

template <class T>
class interval_gpu {
 public:
  __device__ __host__ interval_gpu();
  __device__ __host__ interval_gpu(T const &v);
  __device__ __host__ interval_gpu(T const &l, T const &u);

  __device__ __host__ T const &lower() const;
  __device__ __host__ T const &upper() const;

  static __device__ __host__ interval_gpu empty();

 private:
  T low;
  T up;
};

// Constructors
template <class T>
inline __device__ __host__ interval_gpu<T>::interval_gpu() {}

template <class T>
inline __device__ __host__ interval_gpu<T>::interval_gpu(T const &l, T const &u)
    : low(l), up(u) {}

template <class T>
inline __device__ __host__ interval_gpu<T>::interval_gpu(T const &v)
    : low(v), up(v) {}

template <class T>
inline __device__ __host__ T const &interval_gpu<T>::lower() const {
  return low;
}

template <class T>
inline __device__ __host__ T const &interval_gpu<T>::upper() const {
  return up;
}

template <class T>
inline __device__ __host__ interval_gpu<T> interval_gpu<T>::empty() {
  rounded_arith<T> rnd;
  return interval_gpu<T>(rnd.nan(), rnd.nan());
}

template <class T>
inline __device__ __host__ bool empty(interval_gpu<T> x) {
  T hash = x.lower() + x.upper();
  return (hash != hash);
}

template <class T>
inline __device__ T width(interval_gpu<T> x) {
  if (empty(x)) return 0;

  rounded_arith<T> rnd;
  return rnd.sub_up(x.upper(), x.lower());
}

// Arithmetic operations

// Unary operators
template <class T>
inline __device__ interval_gpu<T> const &operator+(interval_gpu<T> const &x) {
  return x;
}

template <class T>
inline __device__ interval_gpu<T> operator-(interval_gpu<T> const &x) {
  return interval_gpu<T>(-x.upper(), -x.lower());
}

// Binary operators
template <class T>
inline __device__ interval_gpu<T> operator+(interval_gpu<T> const &x,
                                            interval_gpu<T> const &y) {
  rounded_arith<T> rnd;
  return interval_gpu<T>(rnd.add_down(x.lower(), y.lower()),
                         rnd.add_up(x.upper(), y.upper()));
}

template <class T>
inline __device__ interval_gpu<T> operator-(interval_gpu<T> const &x,
                                            interval_gpu<T> const &y) {
  rounded_arith<T> rnd;
  return interval_gpu<T>(rnd.sub_down(x.lower(), y.upper()),
                         rnd.sub_up(x.upper(), y.lower()));
}

inline __device__ float min4(float a, float b, float c, float d) {
  return fminf(fminf(a, b), fminf(c, d));
}

inline __device__ float max4(float a, float b, float c, float d) {
  return fmaxf(fmaxf(a, b), fmaxf(c, d));
}

inline __device__ double min4(double a, double b, double c, double d) {
  return fmin(fmin(a, b), fmin(c, d));
}

inline __device__ double max4(double a, double b, double c, double d) {
  return fmax(fmax(a, b), fmax(c, d));
}

template <class T>
inline __device__ interval_gpu<T> operator*(interval_gpu<T> const &x,
                                            interval_gpu<T> const &y) {
  // Textbook implementation: 14 flops, but no branch.
  rounded_arith<T> rnd;
  return interval_gpu<T>(
      min4(rnd.mul_down(x.lower(), y.lower()),
           rnd.mul_down(x.lower(), y.upper()),
           rnd.mul_down(x.upper(), y.lower()),
           rnd.mul_down(x.upper(), y.upper())),
      max4(rnd.mul_up(x.lower(), y.lower()), rnd.mul_up(x.lower(), y.upper()),
           rnd.mul_up(x.upper(), y.lower()), rnd.mul_up(x.upper(), y.upper())));
}

// Center of an interval
// Typically used for bisection
template <class T>
inline __device__ T median(interval_gpu<T> const &x) {
  rounded_arith<T> rnd;
  return rnd.median(x.lower(), x.upper());
}

// Intersection between two intervals (can be empty)
template <class T>
inline __device__ interval_gpu<T> intersect(interval_gpu<T> const &x,
                                            interval_gpu<T> const &y) {
  rounded_arith<T> rnd;
  T const &l = rnd.max(x.lower(), y.lower());
  T const &u = rnd.min(x.upper(), y.upper());

  if (l <= u)
    return interval_gpu<T>(l, u);
  else
    return interval_gpu<T>::empty();
}

// Division by an interval which does not contain 0.
// GPU-optimized implementation assuming division is expensive
template <class T>
inline __device__ interval_gpu<T> div_non_zero(interval_gpu<T> const &x,
                                               interval_gpu<T> const &y) {
  rounded_arith<T> rnd;
  typedef interval_gpu<T> I;
  T xl, yl, xu, yu;

  if (y.upper() < 0) {
    xl = x.upper();
    xu = x.lower();
  } else {
    xl = x.lower();
    xu = x.upper();
  }

  if (x.upper() < 0) {
    yl = y.lower();
    yu = y.upper();
  } else if (x.lower() < 0) {
    if (y.upper() < 0) {
      yl = y.upper();
      yu = y.upper();
    } else {
      yl = y.lower();
      yu = y.lower();
    }
  } else {
    yl = y.upper();
    yu = y.lower();
  }

  return I(rnd.div_down(xl, yl), rnd.div_up(xu, yu));
}

template <class T>
inline __device__ interval_gpu<T> div_positive(interval_gpu<T> const &x,
                                               T const &yu) {
  // assert(yu > 0);
  if (x.lower() == 0 && x.upper() == 0) return x;

  rounded_arith<T> rnd;
  typedef interval_gpu<T> I;
  const T &xl = x.lower();
  const T &xu = x.upper();

  if (xu < 0)
    return I(rnd.neg_inf(), rnd.div_up(xu, yu));
  else if (xl < 0)
    return I(rnd.neg_inf(), rnd.pos_inf());
  else
    return I(rnd.div_down(xl, yu), rnd.pos_inf());
}

template <class T>
inline __device__ interval_gpu<T> div_negative(interval_gpu<T> const &x,
                                               T const &yl) {
  // assert(yu > 0);
  if (x.lower() == 0 && x.upper() == 0) return x;

  rounded_arith<T> rnd;
  typedef interval_gpu<T> I;
  const T &xl = x.lower();
  const T &xu = x.upper();

  if (xu < 0)
    return I(rnd.div_down(xu, yl), rnd.pos_inf());
  else if (xl < 0)
    return I(rnd.neg_inf(), rnd.pos_inf());
  else
    return I(rnd.neg_inf(), rnd.div_up(xl, yl));
}

template <class T>
inline __device__ interval_gpu<T> div_zero_part1(interval_gpu<T> const &x,
                                                 interval_gpu<T> const &y,
                                                 bool &b) {
  if (x.lower() == 0 && x.upper() == 0) {
    b = false;
    return x;
  }

  rounded_arith<T> rnd;
  typedef interval_gpu<T> I;
  const T &xl = x.lower();
  const T &xu = x.upper();
  const T &yl = y.lower();
  const T &yu = y.upper();

  if (xu < 0) {
    b = true;
    return I(rnd.neg_inf(), rnd.div_up(xu, yu));
  } else if (xl < 0) {
    b = false;
    return I(rnd.neg_inf(), rnd.pos_inf());
  } else {
    b = true;
    return I(rnd.neg_inf(), rnd.div_up(xl, yl));
  }
}

template <class T>
inline __device__ interval_gpu<T> div_zero_part2(interval_gpu<T> const &x,
                                                 interval_gpu<T> const &y) {
  rounded_arith<T> rnd;
  typedef interval_gpu<T> I;
  const T &xl = x.lower();
  const T &xu = x.upper();
  const T &yl = y.lower();
  const T &yu = y.upper();

  if (xu < 0)
    return I(rnd.div_down(xu, yl), rnd.pos_inf());
  else
    return I(rnd.div_down(xl, yu), rnd.pos_inf());
}

template <class T>
inline __device__ interval_gpu<T> division_part1(interval_gpu<T> const &x,
                                                 interval_gpu<T> const &y,
                                                 bool &b) {
  b = false;

  if (y.lower() <= 0 && y.upper() >= 0)
    if (y.lower() != 0)
      if (y.upper() != 0)
        return div_zero_part1(x, y, b);
      else
        return div_negative(x, y.lower());
    else if (y.upper() != 0)
      return div_positive(x, y.upper());
    else
      return interval_gpu<T>::empty();
  else
    return div_non_zero(x, y);
}

template <class T>
inline __device__ interval_gpu<T> division_part2(interval_gpu<T> const &x,
                                                 interval_gpu<T> const &y,
                                                 bool b = true) {
  if (!b) return interval_gpu<T>::empty();

  return div_zero_part2(x, y);
}

template <class T>
inline __device__ interval_gpu<T> square(interval_gpu<T> const &x) {
  typedef interval_gpu<T> I;
  rounded_arith<T> rnd;
  const T &xl = x.lower();
  const T &xu = x.upper();

  if (xl >= 0)
    return I(rnd.mul_down(xl, xl), rnd.mul_up(xu, xu));
  else if (xu <= 0)
    return I(rnd.mul_down(xu, xu), rnd.mul_up(xl, xl));
  else
    return I(static_cast<T>(0),
             rnd.max(rnd.mul_up(xl, xl), rnd.mul_up(xu, xu)));
}

#endif
