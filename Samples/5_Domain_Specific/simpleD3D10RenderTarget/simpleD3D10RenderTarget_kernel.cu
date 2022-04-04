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

/* This example demonstrates how to use the CUDA Direct3D bindings with the
 * runtime API.
 * Device code.
 */

#ifndef SIMPLED3D10RENDERTARGET_KERNEL_CU
#define SIMPLED3D10RENDERTARGET_KERNEL_CU

// includes, C string library
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, cuda
#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>

// includes, project
#include <helper_cuda.h>  // includes cuda.h and cuda_runtime_api.h
//#include "checkCudaErrors"

#define BIN_COUNT 256
#define HISTOGRAM_SIZE (BIN_COUNT * sizeof(unsigned int))

texture<uchar4, 2, cudaReadModeElementType> colorTex;

////////////////////////////////////////////////////////////////////////////////
// GPU-specific definitions
////////////////////////////////////////////////////////////////////////////////
// Fast mul on G8x / G9x / G100
#define IMUL(a, b) __mul24(a, b)

// Machine warp size
// G80's warp size is 32 threads
#define WARP_LOG2SIZE 5

// Warps in thread block for histogram256Kernel()
#define WARP_N 6

// Corresponding thread block size in threads for histogram256Kernel()
#define THREAD_N (WARP_N << WARP_LOG2SIZE)

// Total histogram size (in counters) per thread block for histogram256Kernel()
#define BLOCK_MEMORY (WARP_N * BIN_COUNT)

// Thread block count for histogram256Kernel()
#define BLOCK_N 64

////////////////////////////////////////////////////////////////////////////////
// If threadPos == threadIdx.x, there are always  4-way bank conflicts,
// since each group of 16 threads (half-warp) accesses different bytes,
// but only within 4 shared memory banks. Having shuffled bits of threadIdx.x
// as in histogram64GPU(), each half-warp accesses different shared memory banks
// avoiding any bank conflicts at all.
// Refer to the supplied whitepaper for detailed explanations.
////////////////////////////////////////////////////////////////////////////////
__device__ inline void addData256(volatile unsigned int *s_WarpHist,
                                  unsigned int data, unsigned int threadTag) {
  unsigned int count;

  do {
    count = s_WarpHist[data] & 0x07FFFFFFU;
    count = threadTag | (count + 1);
    s_WarpHist[data] = count;
  } while (s_WarpHist[data] != count);
}

////////////////////////////////////////////////////////////////////////////////
// Main histogram calculation kernel
////////////////////////////////////////////////////////////////////////////////
static __global__ void histogramTex256Kernel(unsigned int *d_Result,
                                             unsigned int width,
                                             unsigned int height, int dataN) {
  // Current global thread index
  const int globalTid = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
  // Total number of threads in the compute grid
  const int numThreads = IMUL(blockDim.x, gridDim.x);

  // Thread tag for addData256()
  // WARP_LOG2SIZE higher bits of counter values are tagged
  // by lower WARP_LOG2SIZE threadID bits
  const unsigned int threadTag = threadIdx.x << (32 - WARP_LOG2SIZE);

  // Shared memory storage for each warp
  volatile __shared__ unsigned int s_Hist[BLOCK_MEMORY];

  // Current warp shared memory base
  const int warpBase = (threadIdx.x >> WARP_LOG2SIZE) * BIN_COUNT;

  // Clear shared memory buffer for current thread block before processing
  for (int pos = threadIdx.x; pos < BLOCK_MEMORY; pos += blockDim.x)
    s_Hist[pos] = 0;

  // Cycle through the entire data set, update subhistograms for each warp
  __syncthreads();

  for (int pos = globalTid; pos < dataN; pos += numThreads) {
    // NOTE: check this... Not sure this is what needs to be done
    int py = pos / width;
    int px = pos - (py * width);
    uchar4 data4 = tex2D(colorTex, px, py);

    addData256(s_Hist + warpBase, (data4.x), threadTag);
    addData256(s_Hist + warpBase, (data4.y), threadTag);
    addData256(s_Hist + warpBase, (data4.z), threadTag);
    addData256(s_Hist + warpBase, (data4.w), threadTag);
  }

  __syncthreads();

  // Merge per-warp histograms into per-block and write to global memory
  for (int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x) {
    unsigned int sum = 0;

    for (int base = 0; base < BLOCK_MEMORY; base += BIN_COUNT)
      sum += s_Hist[base + pos] & 0x07FFFFFFU;

    d_Result[blockIdx.x * BIN_COUNT + pos] = sum;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Merge BLOCK_N subhistograms of BIN_COUNT bins into final histogram
///////////////////////////////////////////////////////////////////////////////
// gridDim.x   == BIN_COUNT
// blockDim.x  == BLOCK_N
// blockIdx.x  == bin counter processed by current block
// threadIdx.x == subhistogram index
static __global__ void mergeHistogramTex256Kernel(unsigned int *d_Result) {
  __shared__ unsigned int data[BLOCK_N];

  // Reads are uncoalesced, but this final stage takes
  // only a fraction of total processing time
  data[threadIdx.x] = d_Result[threadIdx.x * BIN_COUNT + blockIdx.x];

  for (int stride = BLOCK_N / 2; stride > 0; stride >>= 1) {
    __syncthreads();

    if (threadIdx.x < stride) data[threadIdx.x] += data[threadIdx.x + stride];
  }

  if (threadIdx.x == 0) d_Result[blockIdx.x] = data[0];
}

////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU histogram
////////////////////////////////////////////////////////////////////////////////

extern "C" void checkCudaError() {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s.\n", cudaGetErrorString(err));
    exit(2);
  }
}

// Maximum block count for histogram64kernel()
// Limits input data size to 756MB
// const int MAX_BLOCK_N = 16384;

// Internal memory allocation
// const int BLOCK_N2 = 32;

extern "C" void createHistogramTex(unsigned int *h_Result, unsigned int width,
                                   unsigned int height, cudaArray *colorArray) {
  cudaBindTextureToArray(colorTex, colorArray);
  checkCudaError();

  histogramTex256Kernel<<<BLOCK_N, THREAD_N>>>(h_Result, width, height,
                                               width * height / 4);
  checkCudaError();

  mergeHistogramTex256Kernel<<<BIN_COUNT, BLOCK_N>>>(h_Result);
  checkCudaError();

  cudaUnbindTexture(colorTex);
  checkCudaError();

#if 0
    // Dummy fill test
    unsigned int toto[256];

    for (int i=0; i<256; i++)
    {
        toto[i] = i * 100;
    }
    cudaMemcpy(h_Result, toto, HISTOGRAM_SIZE, cudaMemcpyHostToDevice);
#endif
  checkCudaError();
}

extern "C" void bindArrayToTexture(cudaArray *pArray) {}

#endif  // #ifndef SIMPLED3D10RENDERTARGET_KERNEL_CU
