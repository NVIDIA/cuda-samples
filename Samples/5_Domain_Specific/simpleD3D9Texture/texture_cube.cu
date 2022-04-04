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

#define PI 3.1415926536f

/*
 * Paint a 2D surface with a moving bulls-eye pattern.  The "face" parameter
 * selects  between 6 different colors to use.  We will use a different color on
 * each face of a  cube map.
 */
__global__ void cuda_kernel_texture_cube(char *surface, int width, int height,
                                         size_t pitch, int face, float t) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned char *pixel;

  // in the case where, due to quantization into grids, we have
  // more threads than pixels, skip the threads which don't
  // correspond to valid pixels
  if (x >= width || y >= height) return;

  // get a pointer to this pixel
  pixel = (unsigned char *)(surface + y * pitch) + 4 * x;

  // populate it
  float theta_x = (2.0f * x) / width - 1.0f;
  float theta_y = (2.0f * y) / height - 1.0f;
  float theta = 2.0f * PI * sqrt(theta_x * theta_x + theta_y * theta_y);
  unsigned char value = 255 * (0.6f + 0.4f * cos(theta + t));

  pixel[3] = 255;  // alpha

  if (face % 2) {
    pixel[0] =         // blue
        pixel[1] =     // green
        pixel[2] = 0;  // red
    pixel[face / 2] = value;
  } else {
    pixel[0] =             // blue
        pixel[1] =         // green
        pixel[2] = value;  // red
    pixel[face / 2] = 0;
  }
}

extern "C" void cuda_texture_cube(void *surface, int width, int height,
                                  size_t pitch, int face, float t) {
  cudaError_t error = cudaSuccess;

  dim3 Db = dim3(16, 16);  // block dimensions are fixed to be 256 threads
  dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

  cuda_kernel_texture_cube<<<Dg, Db>>>((char *)surface, width, height, pitch,
                                       face, t);

  error = cudaGetLastError();

  if (error != cudaSuccess) {
    printf("cuda_kernel_texture_cube() failed to launch error = %d\n", error);
  }
}
