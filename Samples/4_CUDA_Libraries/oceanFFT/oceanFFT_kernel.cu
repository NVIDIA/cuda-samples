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

///////////////////////////////////////////////////////////////////////////////
#include <cufft.h>
#include <math_constants.h>

// Round a / b to nearest higher integer value
int cuda_iDivUp(int a, int b) { return (a + (b - 1)) / b; }

// complex math functions
__device__ float2 conjugate(float2 arg) { return make_float2(arg.x, -arg.y); }

__device__ float2 complex_exp(float arg) {
  return make_float2(cosf(arg), sinf(arg));
}

__device__ float2 complex_add(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float2 complex_mult(float2 ab, float2 cd) {
  return make_float2(ab.x * cd.x - ab.y * cd.y, ab.x * cd.y + ab.y * cd.x);
}

// generate wave heightfield at time t based on initial heightfield and
// dispersion relationship
__global__ void generateSpectrumKernel(float2 *h0, float2 *ht,
                                       unsigned int in_width,
                                       unsigned int out_width,
                                       unsigned int out_height, float t,
                                       float patchSize) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int in_index = y * in_width + x;
  unsigned int in_mindex =
      (out_height - y) * in_width + (out_width - x);  // mirrored
  unsigned int out_index = y * out_width + x;

  // calculate wave vector
  float2 k;
  k.x = (-(int)out_width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
  k.y = (-(int)out_width / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

  // calculate dispersion w(k)
  float k_len = sqrtf(k.x * k.x + k.y * k.y);
  float w = sqrtf(9.81f * k_len);

  if ((x < out_width) && (y < out_height)) {
    float2 h0_k = h0[in_index];
    float2 h0_mk = h0[in_mindex];

    // output frequency-space complex values
    ht[out_index] =
        complex_add(complex_mult(h0_k, complex_exp(w * t)),
                    complex_mult(conjugate(h0_mk), complex_exp(-w * t)));
    // ht[out_index] = h0_k;
  }
}

// update height map values based on output of FFT
__global__ void updateHeightmapKernel(float *heightMap, float2 *ht,
                                      unsigned int width) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int i = y * width + x;

  // cos(pi * (m1 + m2))
  float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;

  heightMap[i] = ht[i].x * sign_correction;
}

// update height map values based on output of FFT
__global__ void updateHeightmapKernel_y(float *heightMap, float2 *ht,
                                        unsigned int width) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int i = y * width + x;

  // cos(pi * (m1 + m2))
  float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;

  heightMap[i] = ht[i].y * sign_correction;
}

// generate slope by partial differences in spatial domain
__global__ void calculateSlopeKernel(float *h, float2 *slopeOut,
                                     unsigned int width, unsigned int height) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int i = y * width + x;

  float2 slope = make_float2(0.0f, 0.0f);

  if ((x > 0) && (y > 0) && (x < width - 1) && (y < height - 1)) {
    slope.x = h[i + 1] - h[i - 1];
    slope.y = h[i + width] - h[i - width];
  }

  slopeOut[i] = slope;
}

// wrapper functions
extern "C" void cudaGenerateSpectrumKernel(float2 *d_h0, float2 *d_ht,
                                           unsigned int in_width,
                                           unsigned int out_width,
                                           unsigned int out_height,
                                           float animTime, float patchSize) {
  dim3 block(8, 8, 1);
  dim3 grid(cuda_iDivUp(out_width, block.x), cuda_iDivUp(out_height, block.y),
            1);
  generateSpectrumKernel<<<grid, block>>>(d_h0, d_ht, in_width, out_width,
                                          out_height, animTime, patchSize);
}

extern "C" void cudaUpdateHeightmapKernel(float *d_heightMap, float2 *d_ht,
                                          unsigned int width,
                                          unsigned int height, bool autoTest) {
  dim3 block(8, 8, 1);
  dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
  if (autoTest) {
    updateHeightmapKernel_y<<<grid, block>>>(d_heightMap, d_ht, width);
  } else {
    updateHeightmapKernel<<<grid, block>>>(d_heightMap, d_ht, width);
  }
}

extern "C" void cudaCalculateSlopeKernel(float *hptr, float2 *slopeOut,
                                         unsigned int width,
                                         unsigned int height) {
  dim3 block(8, 8, 1);
  dim3 grid2(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
  calculateSlopeKernel<<<grid2, block>>>(hptr, slopeOut, width, height);
}
