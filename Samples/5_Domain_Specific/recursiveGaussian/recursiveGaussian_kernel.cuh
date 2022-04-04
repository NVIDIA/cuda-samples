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
  Recursive Gaussian filter
*/

#ifndef _RECURSIVEGAUSSIAN_KERNEL_CU_
#define _RECURSIVEGAUSSIAN_KERNEL_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <helper_cuda.h>
#include <helper_math.h>

#define BLOCK_DIM 16
#define CLAMP_TO_EDGE 1

// Transpose kernel (see transpose CUDA Sample for details)
__global__ void d_transpose(uint *odata, uint *idata, int width, int height) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  __shared__ uint block[BLOCK_DIM][BLOCK_DIM + 1];

  // read the matrix tile into shared memory
  unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
  unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

  if ((xIndex < width) && (yIndex < height)) {
    unsigned int index_in = yIndex * width + xIndex;
    block[threadIdx.y][threadIdx.x] = idata[index_in];
  }

  cg::sync(cta);

  // write the transposed matrix tile to global memory
  xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
  yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;

  if ((xIndex < height) && (yIndex < width)) {
    unsigned int index_out = yIndex * height + xIndex;
    odata[index_out] = block[threadIdx.x][threadIdx.y];
  }
}

// RGBA version
// reads from 32-bit uint array holding 8-bit RGBA

// convert floating point rgba color to 32-bit integer
__device__ uint rgbaFloatToInt(float4 rgba) {
  rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
  rgba.y = __saturatef(rgba.y);
  rgba.z = __saturatef(rgba.z);
  rgba.w = __saturatef(rgba.w);
  return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) |
         (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

// convert from 32-bit int to float4
__device__ float4 rgbaIntToFloat(uint c) {
  float4 rgba;
  rgba.x = (c & 0xff) / 255.0f;
  rgba.y = ((c >> 8) & 0xff) / 255.0f;
  rgba.z = ((c >> 16) & 0xff) / 255.0f;
  rgba.w = ((c >> 24) & 0xff) / 255.0f;
  return rgba;
}

/*
  simple 1st order recursive filter
  - processes one image column per thread

  parameters:
  id - pointer to input data (RGBA image packed into 32-bit integers)
  od - pointer to output data
  w  - image width
  h  - image height
  a  - blur parameter
*/

__global__ void d_simpleRecursive_rgba(uint *id, uint *od, int w, int h,
                                       float a) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= w) return;

  id += x;  // advance pointers to correct column
  od += x;

  // forward pass
  float4 yp = rgbaIntToFloat(*id);  // previous output

  for (int y = 0; y < h; y++) {
    float4 xc = rgbaIntToFloat(*id);
    float4 yc =
        xc + a * (yp - xc);  // simple lerp between current and previous value
    *od = rgbaFloatToInt(yc);
    id += w;
    od += w;  // move to next row
    yp = yc;
  }

  // reset pointers to point to last element in column
  id -= w;
  od -= w;

  // reverse pass
  // ensures response is symmetrical
  yp = rgbaIntToFloat(*id);

  for (int y = h - 1; y >= 0; y--) {
    float4 xc = rgbaIntToFloat(*id);
    float4 yc = xc + a * (yp - xc);
    *od = rgbaFloatToInt((rgbaIntToFloat(*od) + yc) * 0.5f);
    id -= w;
    od -= w;  // move to previous row
    yp = yc;
  }
}

/*
  recursive Gaussian filter

  parameters:
  id - pointer to input data (RGBA image packed into 32-bit integers)
  od - pointer to output data
  w  - image width
  h  - image height
  a0-a3, b1, b2, coefp, coefn - filter parameters
*/

__global__ void d_recursiveGaussian_rgba(uint *id, uint *od, int w, int h,
                                         float a0, float a1, float a2, float a3,
                                         float b1, float b2, float coefp,
                                         float coefn) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= w) return;

  id += x;  // advance pointers to correct column
  od += x;

  // forward pass
  float4 xp = make_float4(0.0f);  // previous input
  float4 yp = make_float4(0.0f);  // previous output
  float4 yb = make_float4(0.0f);  // previous output by 2
#if CLAMP_TO_EDGE
  xp = rgbaIntToFloat(*id);
  yb = coefp * xp;
  yp = yb;
#endif

  for (int y = 0; y < h; y++) {
    float4 xc = rgbaIntToFloat(*id);
    float4 yc = a0 * xc + a1 * xp - b1 * yp - b2 * yb;
    *od = rgbaFloatToInt(yc);
    id += w;
    od += w;  // move to next row
    xp = xc;
    yb = yp;
    yp = yc;
  }

  // reset pointers to point to last element in column
  id -= w;
  od -= w;

  // reverse pass
  // ensures response is symmetrical
  float4 xn = make_float4(0.0f);
  float4 xa = make_float4(0.0f);
  float4 yn = make_float4(0.0f);
  float4 ya = make_float4(0.0f);
#if CLAMP_TO_EDGE
  xn = xa = rgbaIntToFloat(*id);
  yn = coefn * xn;
  ya = yn;
#endif

  for (int y = h - 1; y >= 0; y--) {
    float4 xc = rgbaIntToFloat(*id);
    float4 yc = a2 * xn + a3 * xa - b1 * yn - b2 * ya;
    xa = xn;
    xn = xc;
    ya = yn;
    yn = yc;
    *od = rgbaFloatToInt(rgbaIntToFloat(*od) + yc);
    id -= w;
    od -= w;  // move to previous row
  }
}

#endif  // #ifndef _GAUSSIAN_KERNEL_H_
