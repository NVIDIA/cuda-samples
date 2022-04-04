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

#ifndef _SIMPLED3D_KERNEL_CU_
#define _SIMPLED3D_KERNEL_CU_

// includes, C string library
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, cuda
#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param pos  pos in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel(float4 *pos, unsigned int width, unsigned int height,
                       float time) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  // calculate uv coordinates
  float u = x / (float)width;
  float v = y / (float)height;
  u = u * 2.0f - 1.0f;
  v = v * 2.0f - 1.0f;

  // calculate simple sine wave pattern
  float freq = 4.0f;
  float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

  // write output vertex
  pos[y * width + x] = make_float4(u, w, v, __int_as_float(0xff00ff00));
}

extern "C" void simpleD3DKernel(float4 *pos, unsigned int width,
                                unsigned int height, float time) {
  cudaError_t error = cudaSuccess;

  dim3 block(8, 8, 1);
  dim3 grid(width / block.x, height / block.y, 1);

  kernel<<<grid, block>>>(pos, width, height, time);

  error = cudaGetLastError();

  if (error != cudaSuccess) {
    printf("kernel() failed to launch error = %d\n", error);
  }
}

#endif  // #ifndef _SIMPLED3D_KERNEL_CU_
