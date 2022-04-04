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
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>
#include "scan_common.h"

// All three kernels run 512 threads per workgroup
// Must be a power of two
#define THREADBLOCK_SIZE 256

////////////////////////////////////////////////////////////////////////////////
// Basic scan codelets
////////////////////////////////////////////////////////////////////////////////
// Naive inclusive scan: O(N * log2(N)) operations
// Allocate 2 * 'size' local memory, initialize the first half
// with 'size' zeros avoiding if(pos >= offset) condition evaluation
// and saving instructions
inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data,
                                      uint size, cg::thread_block cta) {
  uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
  s_Data[pos] = 0;
  pos += size;
  s_Data[pos] = idata;

  for (uint offset = 1; offset < size; offset <<= 1) {
    cg::sync(cta);
    uint t = s_Data[pos] + s_Data[pos - offset];
    cg::sync(cta);
    s_Data[pos] = t;
  }

  return s_Data[pos];
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data,
                                      uint size, cg::thread_block cta) {
  return scan1Inclusive(idata, s_Data, size, cta) - idata;
}

inline __device__ uint4 scan4Inclusive(uint4 idata4, volatile uint *s_Data,
                                       uint size, cg::thread_block cta) {
  // Level-0 inclusive scan
  idata4.y += idata4.x;
  idata4.z += idata4.y;
  idata4.w += idata4.z;

  // Level-1 exclusive scan
  uint oval = scan1Exclusive(idata4.w, s_Data, size / 4, cta);

  idata4.x += oval;
  idata4.y += oval;
  idata4.z += oval;
  idata4.w += oval;

  return idata4;
}

// Exclusive vector scan: the array to be scanned is stored
// in local thread memory scope as uint4
inline __device__ uint4 scan4Exclusive(uint4 idata4, volatile uint *s_Data,
                                       uint size, cg::thread_block cta) {
  uint4 odata4 = scan4Inclusive(idata4, s_Data, size, cta);
  odata4.x -= idata4.x;
  odata4.y -= idata4.y;
  odata4.z -= idata4.z;
  odata4.w -= idata4.w;
  return odata4;
}

////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void scanExclusiveShared(uint4 *d_Dst, uint4 *d_Src, uint size) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

  uint pos = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data
  uint4 idata4 = d_Src[pos];

  // Calculate exclusive scan
  uint4 odata4 = scan4Exclusive(idata4, s_Data, size, cta);

  // Write back
  d_Dst[pos] = odata4;
}

// Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__global__ void scanExclusiveShared2(uint *d_Buf, uint *d_Dst, uint *d_Src,
                                     uint N, uint arrayLength) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

  // Skip loads and stores for inactive threads of last threadblock (pos >= N)
  uint pos = blockIdx.x * blockDim.x + threadIdx.x;

  // Load top elements
  // Convert results of bottom-level scan back to inclusive
  uint idata = 0;

  if (pos < N)
    idata = d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos] +
            d_Src[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];

  // Compute
  uint odata = scan1Exclusive(idata, s_Data, arrayLength, cta);

  // Avoid out-of-bound access
  if (pos < N) {
    d_Buf[pos] = odata;
  }
}

// Final step of large-array scan: combine basic inclusive scan with exclusive
// scan of top elements of input arrays
__global__ void uniformUpdate(uint4 *d_Data, uint *d_Buffer) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ uint buf;
  uint pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0) {
    buf = d_Buffer[blockIdx.x];
  }

  cg::sync(cta);

  uint4 data4 = d_Data[pos];
  data4.x += buf;
  data4.y += buf;
  data4.z += buf;
  data4.w += buf;
  d_Data[pos] = data4;
}

////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
// Derived as 32768 (max power-of-two gridDim.x) * 4 * THREADBLOCK_SIZE
// Due to scanExclusiveShared<<<>>>() 1D block addressing
extern "C" const uint MAX_BATCH_ELEMENTS = 64 * 1048576;
extern "C" const uint MIN_SHORT_ARRAY_SIZE = 4;
extern "C" const uint MAX_SHORT_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;
extern "C" const uint MIN_LARGE_ARRAY_SIZE = 8 * THREADBLOCK_SIZE;
extern "C" const uint MAX_LARGE_ARRAY_SIZE =
    4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE;

// Internal exclusive scan buffer
static uint *d_Buf;

extern "C" void initScan(void) {
  checkCudaErrors(
      cudaMalloc((void **)&d_Buf,
                 (MAX_BATCH_ELEMENTS / (4 * THREADBLOCK_SIZE)) * sizeof(uint)));
}

extern "C" void closeScan(void) { checkCudaErrors(cudaFree(d_Buf)); }

static uint factorRadix2(uint &log2L, uint L) {
  if (!L) {
    log2L = 0;
    return 0;
  } else {
    for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++)
      ;

    return L;
  }
}

static uint iDivUp(uint dividend, uint divisor) {
  return ((dividend % divisor) == 0) ? (dividend / divisor)
                                     : (dividend / divisor + 1);
}

extern "C" size_t scanExclusiveShort(uint *d_Dst, uint *d_Src, uint batchSize,
                                     uint arrayLength) {
  // Check power-of-two factorization
  uint log2L;
  uint factorizationRemainder = factorRadix2(log2L, arrayLength);
  assert(factorizationRemainder == 1);

  // Check supported size range
  assert((arrayLength >= MIN_SHORT_ARRAY_SIZE) &&
         (arrayLength <= MAX_SHORT_ARRAY_SIZE));

  // Check total batch size limit
  assert((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS);

  // Check all threadblocks to be fully packed with data
  assert((batchSize * arrayLength) % (4 * THREADBLOCK_SIZE) == 0);

  scanExclusiveShared<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE),
                        THREADBLOCK_SIZE>>>((uint4 *)d_Dst, (uint4 *)d_Src,
                                            arrayLength);
  getLastCudaError("scanExclusiveShared() execution FAILED\n");

  return THREADBLOCK_SIZE;
}

extern "C" size_t scanExclusiveLarge(uint *d_Dst, uint *d_Src, uint batchSize,
                                     uint arrayLength) {
  // Check power-of-two factorization
  uint log2L;
  uint factorizationRemainder = factorRadix2(log2L, arrayLength);
  assert(factorizationRemainder == 1);

  // Check supported size range
  assert((arrayLength >= MIN_LARGE_ARRAY_SIZE) &&
         (arrayLength <= MAX_LARGE_ARRAY_SIZE));

  // Check total batch size limit
  assert((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS);

  scanExclusiveShared<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE),
                        THREADBLOCK_SIZE>>>((uint4 *)d_Dst, (uint4 *)d_Src,
                                            4 * THREADBLOCK_SIZE);
  getLastCudaError("scanExclusiveShared() execution FAILED\n");

  // Not all threadblocks need to be packed with input data:
  // inactive threads of highest threadblock just don't do global reads and
  // writes
  const uint blockCount2 = iDivUp(
      (batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE);
  scanExclusiveShared2<<<blockCount2, THREADBLOCK_SIZE>>>(
      (uint *)d_Buf, (uint *)d_Dst, (uint *)d_Src,
      (batchSize * arrayLength) / (4 * THREADBLOCK_SIZE),
      arrayLength / (4 * THREADBLOCK_SIZE));
  getLastCudaError("scanExclusiveShared2() execution FAILED\n");

  uniformUpdate<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE),
                  THREADBLOCK_SIZE>>>((uint4 *)d_Dst, (uint *)d_Buf);
  getLastCudaError("uniformUpdate() execution FAILED\n");

  return THREADBLOCK_SIZE;
}
