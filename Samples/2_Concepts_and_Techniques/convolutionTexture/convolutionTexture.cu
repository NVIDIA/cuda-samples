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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>

#include "convolutionTexture_common.h"

////////////////////////////////////////////////////////////////////////////////
// GPU-specific defines
////////////////////////////////////////////////////////////////////////////////
// Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) (__mul24((a), (b)) + (c))

// Use unrolled innermost convolution loop
#define UNROLL_INNER 1

// Round a / b to nearest higher integer value
inline int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel and input array storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[KERNEL_LENGTH];

extern "C" void setConvolutionKernel(float *h_Kernel) {
  cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
}

////////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
////////////////////////////////////////////////////////////////////////////////
template <int i>
__device__ float convolutionRow(float x, float y, cudaTextureObject_t texSrc) {
  return tex2D<float>(texSrc, x + (float)(KERNEL_RADIUS - i), y) * c_Kernel[i] +
         convolutionRow<i - 1>(x, y, texSrc);
}

template <>
__device__ float convolutionRow<-1>(float x, float y,
                                    cudaTextureObject_t texSrc) {
  return 0;
}

template <int i>
__device__ float convolutionColumn(float x, float y,
                                   cudaTextureObject_t texSrc) {
  return tex2D<float>(texSrc, x, y + (float)(KERNEL_RADIUS - i)) * c_Kernel[i] +
         convolutionColumn<i - 1>(x, y, texSrc);
}

template <>
__device__ float convolutionColumn<-1>(float x, float y,
                                       cudaTextureObject_t texSrc) {
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowsKernel(float *d_Dst, int imageW, int imageH,
                                      cudaTextureObject_t texSrc) {
  const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
  const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
  const float x = (float)ix + 0.5f;
  const float y = (float)iy + 0.5f;

  if (ix >= imageW || iy >= imageH) {
    return;
  }

  float sum = 0;

#if (UNROLL_INNER)
  sum = convolutionRow<2 * KERNEL_RADIUS>(x, y, texSrc);
#else

  for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
    sum += tex2D<float>(texSrc, x + (float)k, y) * c_Kernel[KERNEL_RADIUS - k];
  }

#endif

  d_Dst[IMAD(iy, imageW, ix)] = sum;
}

extern "C" void convolutionRowsGPU(float *d_Dst, cudaArray *a_Src, int imageW,
                                   int imageH, cudaTextureObject_t texSrc) {
  dim3 threads(16, 12);
  dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

  convolutionRowsKernel<<<blocks, threads>>>(d_Dst, imageW, imageH, texSrc);
  getLastCudaError("convolutionRowsKernel() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnsKernel(float *d_Dst, int imageW, int imageH,
                                         cudaTextureObject_t texSrc) {
  const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
  const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
  const float x = (float)ix + 0.5f;
  const float y = (float)iy + 0.5f;

  if (ix >= imageW || iy >= imageH) {
    return;
  }

  float sum = 0;

#if (UNROLL_INNER)
  sum = convolutionColumn<2 * KERNEL_RADIUS>(x, y, texSrc);
#else

  for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
    sum += tex2D<float>(texSrc, x, y + (float)k) * c_Kernel[KERNEL_RADIUS - k];
  }

#endif

  d_Dst[IMAD(iy, imageW, ix)] = sum;
}

extern "C" void convolutionColumnsGPU(float *d_Dst, cudaArray *a_Src,
                                      int imageW, int imageH,
                                      cudaTextureObject_t texSrc) {
  dim3 threads(16, 12);
  dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

  convolutionColumnsKernel<<<blocks, threads>>>(d_Dst, imageW, imageH, texSrc);
  getLastCudaError("convolutionColumnsKernel() execution failed\n");
}
