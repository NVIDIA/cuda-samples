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

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>
#include <assert.h>
#include "mergeSort_common.h"

inline __device__ void Comparator(uint &keyA, uint &valA, uint &keyB,
                                  uint &valB, uint arrowDir) {
  uint t;

  if ((keyA > keyB) == arrowDir) {
    t = keyA;
    keyA = keyB;
    keyB = t;
    t = valA;
    valA = valB;
    valB = t;
  }
}

__global__ void bitonicSortSharedKernel(uint *d_DstKey, uint *d_DstVal,
                                        uint *d_SrcKey, uint *d_SrcVal,
                                        uint arrayLength, uint sortDir) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Shared memory storage for one or more short vectors
  __shared__ uint s_key[SHARED_SIZE_LIMIT];
  __shared__ uint s_val[SHARED_SIZE_LIMIT];

  // Offset to the beginning of subbatch and load data
  d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  s_key[threadIdx.x + 0] = d_SrcKey[0];
  s_val[threadIdx.x + 0] = d_SrcVal[0];
  s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
      d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
  s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
      d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

  for (uint size = 2; size < arrayLength; size <<= 1) {
    // Bitonic merge
    uint dir = (threadIdx.x & (size / 2)) != 0;

    for (uint stride = size / 2; stride > 0; stride >>= 1) {
      cg::sync(cta);
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                 s_val[pos + stride], dir);
    }
  }

  // ddd == sortDir for the last bitonic merge step
  {
    for (uint stride = arrayLength / 2; stride > 0; stride >>= 1) {
      cg::sync(cta);
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                 s_val[pos + stride], sortDir);
    }
  }

  cg::sync(cta);
  d_DstKey[0] = s_key[threadIdx.x + 0];
  d_DstVal[0] = s_val[threadIdx.x + 0];
  d_DstKey[(SHARED_SIZE_LIMIT / 2)] =
      s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
  d_DstVal[(SHARED_SIZE_LIMIT / 2)] =
      s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

// Helper function (also used by odd-even merge sort)
extern "C" uint factorRadix2(uint *log2L, uint L) {
  if (!L) {
    *log2L = 0;
    return 0;
  } else {
    for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++)
      ;

    return L;
  }
}

extern "C" void bitonicSortShared(uint *d_DstKey, uint *d_DstVal,
                                  uint *d_SrcKey, uint *d_SrcVal,
                                  uint batchSize, uint arrayLength,
                                  uint sortDir) {
  // Nothing to sort
  if (arrayLength < 2) {
    return;
  }

  // Only power-of-two array lengths are supported by this implementation
  uint log2L;
  uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
  assert(factorizationRemainder == 1);

  uint blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
  uint threadCount = SHARED_SIZE_LIMIT / 2;

  assert(arrayLength <= SHARED_SIZE_LIMIT);
  assert((batchSize * arrayLength) % SHARED_SIZE_LIMIT == 0);

  bitonicSortSharedKernel<<<blockCount, threadCount>>>(
      d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, sortDir);
  getLastCudaError("bitonicSortSharedKernel<<<>>> failed!\n");
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 3: merge elementary intervals
////////////////////////////////////////////////////////////////////////////////
static inline __host__ __device__ uint iDivUp(uint a, uint b) {
  return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

static inline __host__ __device__ uint getSampleCount(uint dividend) {
  return iDivUp(dividend, SAMPLE_STRIDE);
}

template <uint sortDir>
static inline __device__ void ComparatorExtended(uint &keyA, uint &valA,
                                                 uint &flagA, uint &keyB,
                                                 uint &valB, uint &flagB,
                                                 uint arrowDir) {
  uint t;

  if ((!(flagA || flagB) && ((keyA > keyB) == arrowDir)) ||
      ((arrowDir == sortDir) && (flagA == 1)) ||
      ((arrowDir != sortDir) && (flagB == 1))) {
    t = keyA;
    keyA = keyB;
    keyB = t;
    t = valA;
    valA = valB;
    valB = t;
    t = flagA;
    flagA = flagB;
    flagB = t;
  }
}

template <uint sortDir>
__global__ void bitonicMergeElementaryIntervalsKernel(
    uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal,
    uint *d_LimitsA, uint *d_LimitsB, uint stride, uint N) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ uint s_key[2 * SAMPLE_STRIDE];
  __shared__ uint s_val[2 * SAMPLE_STRIDE];
  __shared__ uint s_inf[2 * SAMPLE_STRIDE];

  const uint intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1);
  const uint segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE;
  d_SrcKey += segmentBase;
  d_SrcVal += segmentBase;
  d_DstKey += segmentBase;
  d_DstVal += segmentBase;

  // Set up threadblock-wide parameters
  __shared__ uint startSrcA, lenSrcA, startSrcB, lenSrcB, startDst;

  if (threadIdx.x == 0) {
    uint segmentElementsA = stride;
    uint segmentElementsB = umin(stride, N - segmentBase - stride);
    uint segmentSamplesA = stride / SAMPLE_STRIDE;
    uint segmentSamplesB = getSampleCount(segmentElementsB);
    uint segmentSamples = segmentSamplesA + segmentSamplesB;

    startSrcA = d_LimitsA[blockIdx.x];
    startSrcB = d_LimitsB[blockIdx.x];
    startDst = startSrcA + startSrcB;

    uint endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1]
                                                    : segmentElementsA;
    uint endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1]
                                                    : segmentElementsB;
    lenSrcA = endSrcA - startSrcA;
    lenSrcB = endSrcB - startSrcB;
  }

  s_inf[threadIdx.x + 0] = 1;
  s_inf[threadIdx.x + SAMPLE_STRIDE] = 1;

  // Load input data
  cg::sync(cta);

  if (threadIdx.x < lenSrcA) {
    s_key[threadIdx.x] = d_SrcKey[0 + startSrcA + threadIdx.x];
    s_val[threadIdx.x] = d_SrcVal[0 + startSrcA + threadIdx.x];
    s_inf[threadIdx.x] = 0;
  }

  // Prepare for bitonic merge by inversing the ordering
  if (threadIdx.x < lenSrcB) {
    s_key[2 * SAMPLE_STRIDE - 1 - threadIdx.x] =
        d_SrcKey[stride + startSrcB + threadIdx.x];
    s_val[2 * SAMPLE_STRIDE - 1 - threadIdx.x] =
        d_SrcVal[stride + startSrcB + threadIdx.x];
    s_inf[2 * SAMPLE_STRIDE - 1 - threadIdx.x] = 0;
  }

  //"Extended" bitonic merge
  for (uint stride = SAMPLE_STRIDE; stride > 0; stride >>= 1) {
    cg::sync(cta);
    uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    ComparatorExtended<sortDir>(s_key[pos + 0], s_val[pos + 0], s_inf[pos + 0],
                                s_key[pos + stride], s_val[pos + stride],
                                s_inf[pos + stride], sortDir);
  }

  // Store sorted data
  cg::sync(cta);
  d_DstKey += startDst;
  d_DstVal += startDst;

  if (threadIdx.x < lenSrcA) {
    d_DstKey[threadIdx.x] = s_key[threadIdx.x];
    d_DstVal[threadIdx.x] = s_val[threadIdx.x];
  }

  if (threadIdx.x < lenSrcB) {
    d_DstKey[lenSrcA + threadIdx.x] = s_key[lenSrcA + threadIdx.x];
    d_DstVal[lenSrcA + threadIdx.x] = s_val[lenSrcA + threadIdx.x];
  }
}

extern "C" void bitonicMergeElementaryIntervals(uint *d_DstKey, uint *d_DstVal,
                                                uint *d_SrcKey, uint *d_SrcVal,
                                                uint *d_LimitsA,
                                                uint *d_LimitsB, uint stride,
                                                uint N, uint sortDir) {
  uint lastSegmentElements = N % (2 * stride);

  uint mergePairs = (lastSegmentElements > stride)
                        ? getSampleCount(N)
                        : (N - lastSegmentElements) / SAMPLE_STRIDE;

  if (sortDir) {
    bitonicMergeElementaryIntervalsKernel<1U><<<mergePairs, SAMPLE_STRIDE>>>(
        d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, d_LimitsA, d_LimitsB, stride,
        N);
    getLastCudaError("mergeElementaryIntervalsKernel<1> failed\n");
  } else {
    bitonicMergeElementaryIntervalsKernel<0U><<<mergePairs, SAMPLE_STRIDE>>>(
        d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, d_LimitsA, d_LimitsB, stride,
        N);
    getLastCudaError("mergeElementaryIntervalsKernel<0> failed\n");
  }
}
