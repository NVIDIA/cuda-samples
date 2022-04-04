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

#ifndef MONTECARLO_REDUCTION_CUH
#define MONTECARLO_REDUCTION_CUH

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////
// This function calculates total sum for each of the two input arrays.
// SUM_N must be power of two
// Unrolling provides a bit of a performance improvement for small
// to medium path counts.
////////////////////////////////////////////////////////////////////////////////

template <class T, int SUM_N, int blockSize>
__device__ void sumReduce(T *sum, T *sum2, cg::thread_block &cta,
                          cg::thread_block_tile<32> &tile32,
                          __TOptionValue *d_CallValue) {
  const int VEC = 32;
  const int tid = cta.thread_rank();

  T beta = sum[tid];
  T beta2 = sum2[tid];
  T temp, temp2;

  for (int i = VEC / 2; i > 0; i >>= 1) {
    if (tile32.thread_rank() < i) {
      temp = sum[tid + i];
      temp2 = sum2[tid + i];
      beta += temp;
      beta2 += temp2;
      sum[tid] = beta;
      sum2[tid] = beta2;
    }
    cg::sync(tile32);
  }
  cg::sync(cta);

  if (tid == 0) {
    beta = 0;
    beta2 = 0;
    for (int i = 0; i < blockDim.x; i += VEC) {
      beta += sum[i];
      beta2 += sum2[i];
    }
    __TOptionValue t = {beta, beta2};
    *d_CallValue = t;
  }
  cg::sync(cta);
}

#endif
