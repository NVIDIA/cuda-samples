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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>
#include "histogram_common.h"

////////////////////////////////////////////////////////////////////////////////
// Shortcut shared memory atomic addition functions
////////////////////////////////////////////////////////////////////////////////

#define TAG_MASK 0xFFFFFFFFU
inline __device__ void addByte(uint *s_WarpHist, uint data, uint threadTag) {
  atomicAdd(s_WarpHist + data, 1);
}

inline __device__ void addWord(uint *s_WarpHist, uint data, uint tag) {
  addByte(s_WarpHist, (data >> 0) & 0xFFU, tag);
  addByte(s_WarpHist, (data >> 8) & 0xFFU, tag);
  addByte(s_WarpHist, (data >> 16) & 0xFFU, tag);
  addByte(s_WarpHist, (data >> 24) & 0xFFU, tag);
}

__global__ void histogram256Kernel(uint *d_PartialHistograms, uint *d_Data,
                                   uint dataCount) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Per-warp subhistogram storage
  __shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
  uint *s_WarpHist =
      s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

// Clear shared memory storage for current threadblock before processing
#pragma unroll

  for (uint i = 0;
       i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE);
       i++) {
    s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
  }

  // Cycle through the entire data set, update subhistograms for each warp
  const uint tag = threadIdx.x << (UINT_BITS - LOG2_WARP_SIZE);

  cg::sync(cta);

  for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount;
       pos += UMUL(blockDim.x, gridDim.x)) {
    uint data = d_Data[pos];
    addWord(s_WarpHist, data, tag);
  }

  // Merge per-warp histograms into per-block and write to global memory
  cg::sync(cta);

  for (uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT;
       bin += HISTOGRAM256_THREADBLOCK_SIZE) {
    uint sum = 0;

    for (uint i = 0; i < WARP_COUNT; i++) {
      sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT] & TAG_MASK;
    }

    d_PartialHistograms[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Merge histogram256() output
// Run one threadblock per bin; each threadblock adds up the same bin counter
// from every partial histogram. Reads are uncoalesced, but mergeHistogram256
// takes only a fraction of total processing time
////////////////////////////////////////////////////////////////////////////////
#define MERGE_THREADBLOCK_SIZE 256

__global__ void mergeHistogram256Kernel(uint *d_Histogram,
                                        uint *d_PartialHistograms,
                                        uint histogramCount) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  uint sum = 0;

  for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE) {
    sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM256_BIN_COUNT];
  }

  __shared__ uint data[MERGE_THREADBLOCK_SIZE];
  data[threadIdx.x] = sum;

  for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    cg::sync(cta);

    if (threadIdx.x < stride) {
      data[threadIdx.x] += data[threadIdx.x + stride];
    }
  }

  if (threadIdx.x == 0) {
    d_Histogram[blockIdx.x] = data[0];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU histogram
////////////////////////////////////////////////////////////////////////////////
// histogram256kernel() intermediate results buffer
static const uint PARTIAL_HISTOGRAM256_COUNT = 240;
static uint *d_PartialHistograms;

// Internal memory allocation
extern "C" void initHistogram256(void) {
  checkCudaErrors(cudaMalloc(
      (void **)&d_PartialHistograms,
      PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(uint)));
}

// Internal memory deallocation
extern "C" void closeHistogram256(void) {
  checkCudaErrors(cudaFree(d_PartialHistograms));
}

extern "C" void histogram256(uint *d_Histogram, void *d_Data, uint byteCount) {
  assert(byteCount % sizeof(uint) == 0);
  histogram256Kernel<<<PARTIAL_HISTOGRAM256_COUNT,
                       HISTOGRAM256_THREADBLOCK_SIZE>>>(
      d_PartialHistograms, (uint *)d_Data, byteCount / sizeof(uint));
  getLastCudaError("histogram256Kernel() execution failed\n");

  mergeHistogram256Kernel<<<HISTOGRAM256_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>(
      d_Histogram, d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT);
  getLastCudaError("mergeHistogram256Kernel() execution failed\n");
}
