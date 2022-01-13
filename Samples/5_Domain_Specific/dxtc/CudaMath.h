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

// Math functions and operators to be used with vector types.

#ifndef CUDAMATH_H
#define CUDAMATH_H

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Use power method to find the first eigenvector.
// https://en.wikipedia.org/wiki/Power_iteration
inline __device__ __host__ float3 firstEigenVector(float matrix[6]) {
  // 8 iterations seems to be more than enough.

  float3 v = make_float3(1.0f, 1.0f, 1.0f);

  for (int i = 0; i < 8; i++) {
    float x = v.x * matrix[0] + v.y * matrix[1] + v.z * matrix[2];
    float y = v.x * matrix[1] + v.y * matrix[3] + v.z * matrix[4];
    float z = v.x * matrix[2] + v.y * matrix[4] + v.z * matrix[5];
    float m = max(max(x, y), z);
    float iv = 1.0f / m;
    v = make_float3(x * iv, y * iv, z * iv);
  }

  return v;
}

inline __device__ void colorSums(const float3 *colors, float3 *sums,
                                 cg::thread_group tile) {
  const int idx = threadIdx.x;

  sums[idx] = colors[idx];
  cg::sync(tile);
  sums[idx] += sums[idx ^ 8];
  cg::sync(tile);
  sums[idx] += sums[idx ^ 4];
  cg::sync(tile);
  sums[idx] += sums[idx ^ 2];
  cg::sync(tile);
  sums[idx] += sums[idx ^ 1];
}

inline __device__ float3 bestFitLine(const float3 *colors, float3 color_sum,
                                     cg::thread_group tile) {
  // Compute covariance matrix of the given colors.
  const int idx = threadIdx.x;

  float3 diff = colors[idx] - color_sum * (1.0f / 16.0f);

  // @@ Eliminate two-way bank conflicts here.
  // @@ It seems that doing that and unrolling the reduction doesn't help...
  __shared__ float covariance[16 * 6];

  covariance[6 * idx + 0] = diff.x * diff.x;  // 0, 6, 12, 2, 8, 14, 4, 10, 0
  covariance[6 * idx + 1] = diff.x * diff.y;
  covariance[6 * idx + 2] = diff.x * diff.z;
  covariance[6 * idx + 3] = diff.y * diff.y;
  covariance[6 * idx + 4] = diff.y * diff.z;
  covariance[6 * idx + 5] = diff.z * diff.z;

  cg::sync(tile);
  for (int d = 8; d > 0; d >>= 1) {
    if (idx < d) {
      covariance[6 * idx + 0] += covariance[6 * (idx + d) + 0];
      covariance[6 * idx + 1] += covariance[6 * (idx + d) + 1];
      covariance[6 * idx + 2] += covariance[6 * (idx + d) + 2];
      covariance[6 * idx + 3] += covariance[6 * (idx + d) + 3];
      covariance[6 * idx + 4] += covariance[6 * (idx + d) + 4];
      covariance[6 * idx + 5] += covariance[6 * (idx + d) + 5];
    }
    cg::sync(tile);
  }

  // Compute first eigen vector.
  return firstEigenVector(covariance);
}

#endif  // CUDAMATH_H
