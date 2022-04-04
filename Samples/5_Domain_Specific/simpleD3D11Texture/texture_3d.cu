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

/*
 * Paint a 3D texture with a gradient in X (blue) and Z (green), and have every
 * other Z slice have full red.
 */
__global__ void cuda_kernel_texture_3d(unsigned char *surface, int width,
                                       int height, int depth, size_t pitch,
                                       size_t pitchSlice, float t) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // in the case where, due to quantization into grids, we have
  // more threads than pixels, skip the threads which don't
  // correspond to valid pixels
  if (x >= width || y >= height) return;

  // walk across the Z slices of this texture.  it should be noted that
  // this is far from optimal data access.
  for (int z = 0; z < depth; ++z) {
    // get a pointer to this pixel
    unsigned char *pixel = surface + z * pitchSlice + y * pitch + 4 * x;
    pixel[0] =
        (unsigned char)(255.f * (0.5f + 0.5f * 
        cos(t + (x * x + y * y + z * z) * 0.0001f * 3.14f)));  // red
    pixel[1] =
        (unsigned char)(255.f * (0.5f + 0.5f * 
        sin(t + (x * x + y * y + z * z) * 0.0001f * 3.14f)));  // green
    pixel[2] = (unsigned char)0;                               // blue
    pixel[3] = 255;                                            // alpha
  }
}

extern "C" void cuda_texture_3d(void *surface, int width, int height, int depth,
                                size_t pitch, size_t pitchSlice, float t) {
  cudaError_t error = cudaSuccess;

  dim3 Db = dim3(16, 16);  // block dimensions are fixed to be 256 threads
  dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

  cuda_kernel_texture_3d<<<Dg, Db>>>((unsigned char *)surface, width, height,
                                     depth, pitch, pitchSlice, t);

  error = cudaGetLastError();

  if (error != cudaSuccess) {
    printf("cuda_kernel_texture_3d() failed to launch error = %d\n", error);
  }
}
