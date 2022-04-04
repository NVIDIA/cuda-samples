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

// Type-specific implementation of rounded arithmetic operators.
// Thin layer over the CUDA intrinsics.

#ifndef CUDA_INTERVAL_ROUNDED_ARITH_H
#define CUDA_INTERVAL_ROUNDED_ARITH_H

// Generic class, no actual implementation yet
template <class T>
struct rounded_arith {
  __device__ T add_down(const T &x, const T &y);
  __device__ T add_up(const T &x, const T &y);
  __device__ T sub_down(const T &x, const T &y);
  __device__ T sub_up(const T &x, const T &y);
  __device__ T mul_down(const T &x, const T &y);
  __device__ T mul_up(const T &x, const T &y);
  __device__ T div_down(const T &x, const T &y);
  __device__ T div_up(const T &x, const T &y);
  __device__ T median(const T &x, const T &y);
  __device__ T sqrt_down(const T &x);
  __device__ T sqrt_up(const T &x);
  __device__ T int_down(const T &x);
  __device__ T int_up(const T &x);

  __device__ T pos_inf();
  __device__ T neg_inf();
  __device__ __host__ T nan();
  __device__ T min(T const &x, T const &y);
  __device__ T max(T const &x, T const &y);
};

// Specialization for float
template <>
struct rounded_arith<float> {
  __device__ float add_down(const float &x, const float &y) {
    return __fadd_rd(x, y);
  }

  __device__ float add_up(const float &x, const float &y) {
    return __fadd_ru(x, y);
  }

  __device__ float sub_down(const float &x, const float &y) {
    return __fadd_rd(x, -y);
  }

  __device__ float sub_up(const float &x, const float &y) {
    return __fadd_ru(x, -y);
  }

  __device__ float mul_down(const float &x, const float &y) {
    return __fmul_rd(x, y);
  }

  __device__ float mul_up(const float &x, const float &y) {
    return __fmul_ru(x, y);
  }

  __device__ float div_down(const float &x, const float &y) {
    return __fdiv_rd(x, y);
  }

  __device__ float div_up(const float &x, const float &y) {
    return __fdiv_ru(x, y);
  }

  __device__ float median(const float &x, const float &y) {
    return (x + y) * .5f;
  }

  __device__ float sqrt_down(const float &x) { return __fsqrt_rd(x); }

  __device__ float sqrt_up(const float &x) { return __fsqrt_ru(x); }

  __device__ float int_down(const float &x) { return floorf(x); }

  __device__ float int_up(const float &x) { return ceilf(x); }

  __device__ float neg_inf() { return __int_as_float(0xff800000); }

  __device__ float pos_inf() { return __int_as_float(0x7f800000); }

  __device__ __host__ float nan() { return nanf(""); }

  __device__ float min(float const &x, float const &y) { return fminf(x, y); }

  __device__ float max(float const &x, float const &y) { return fmaxf(x, y); }
};

// Specialization for double
template <>
struct rounded_arith<double> {
  __device__ double add_down(const double &x, const double &y) {
    return __dadd_rd(x, y);
  }

  __device__ double add_up(const double &x, const double &y) {
    return __dadd_ru(x, y);
  }

  __device__ double sub_down(const double &x, const double &y) {
    return __dadd_rd(x, -y);
  }

  __device__ double sub_up(const double &x, const double &y) {
    return __dadd_ru(x, -y);
  }

  __device__ double mul_down(const double &x, const double &y) {
    return __dmul_rd(x, y);
  }

  __device__ double mul_up(const double &x, const double &y) {
    return __dmul_ru(x, y);
  }

  __device__ double div_down(const double &x, const double &y) {
    return __ddiv_rd(x, y);
  }

  __device__ double div_up(const double &x, const double &y) {
    return __ddiv_ru(x, y);
  }
  __device__ double median(const double &x, const double &y) {
    return (x + y) * .5;
  }

  __device__ double sqrt_down(const double &x) { return __dsqrt_rd(x); }

  __device__ double sqrt_up(const double &x) { return __dsqrt_ru(x); }

  __device__ double int_down(const double &x) { return floor(x); }

  __device__ double int_up(const double &x) { return ceil(x); }

  __device__ double neg_inf() {
    return __longlong_as_double(0xfff0000000000000ull);
  }

  __device__ double pos_inf() {
    return __longlong_as_double(0x7ff0000000000000ull);
  }
  __device__ __host__ double nan() { return ::nan(""); }

  __device__ double min(double const &x, double const &y) { return fmin(x, y); }

  __device__ double max(double const &x, double const &y) { return fmax(x, y); }
};

#endif
