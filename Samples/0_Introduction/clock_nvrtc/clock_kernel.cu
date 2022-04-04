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

/*
 * This example shows how to use the clock function to measure the performance
 * of block of threads of a kernel accurately. Blocks are executed in parallel
 * and out of order. Since there's no synchronization mechanism between blocks,
 * we measure the clock once for each block. The clock samples are written to
 * device memory.
 */

// This kernel computes a standard parallel reduction and evaluates the
// time it takes to do that for each block. The timing results are stored
// in device memory.

extern "C" __global__ void timedReduction(const float *input, float *output,
                                          clock_t *timer) {
  // __shared__ float shared[2 * blockDim.x];
  extern __shared__ float shared[];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  if (tid == 0) timer[bid] = clock();

  // Copy input.
  shared[tid] = input[tid];
  shared[tid + blockDim.x] = input[tid + blockDim.x];

  // Perform reduction to find minimum.
  for (int d = blockDim.x; d > 0; d /= 2) {
    __syncthreads();

    if (tid < d) {
      float f0 = shared[tid];
      float f1 = shared[tid + d];

      if (f1 < f0) {
        shared[tid] = f1;
      }
    }
  }

  // Write result.
  if (tid == 0) output[bid] = shared[0];

  __syncthreads();

  if (tid == 0) timer[bid + gridDim.x] = clock();
}
