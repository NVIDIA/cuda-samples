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
 * This sample demonstrates two adaptive image denoising techniques:
 * KNN and NLM, based on computation of both geometric and color distance
 * between texels. While both techniques are already implemented in the
 * DirectX SDK using shaders, massively speeded up variation
 * of the latter technique, taking advantage of shared memory, is implemented
 * in addition to DirectX counterparts.
 * See supplied whitepaper for more explanations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include "imageDenoising.h"

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
float Max(float x, float y) { return (x > y) ? x : y; }

float Min(float x, float y) { return (x < y) ? x : y; }

int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

__device__ float lerpf(float a, float b, float c) { return a + (b - a) * c; }

__device__ float vecLen(float4 a, float4 b) {
  return ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) +
          (b.z - a.z) * (b.z - a.z));
}

__device__ TColor make_color(float r, float g, float b, float a) {
  return ((int)(a * 255.0f) << 24) | ((int)(b * 255.0f) << 16) |
         ((int)(g * 255.0f) << 8) | ((int)(r * 255.0f) << 0);
}

////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
// Texture object and channel descriptor for image texture
cudaTextureObject_t texImage;
cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();

// CUDA array descriptor
cudaArray *a_Src;

////////////////////////////////////////////////////////////////////////////////
// Filtering kernels
////////////////////////////////////////////////////////////////////////////////
#include "imageDenoising_copy_kernel.cuh"
#include "imageDenoising_knn_kernel.cuh"
#include "imageDenoising_nlm_kernel.cuh"
#include "imageDenoising_nlm2_kernel.cuh"

extern "C" cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW,
                                        int imageH) {
  cudaError_t error;

  error = cudaMallocArray(&a_Src, &uchar4tex, imageW, imageH);
  error = cudaMemcpy2DToArray(a_Src, 0, 0, *h_Src, sizeof(uchar4) * imageW,
                              sizeof(uchar4) * imageW, imageH,
                              cudaMemcpyHostToDevice);

  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = a_Src;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.addressMode[1] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeNormalizedFloat;

  checkCudaErrors(cudaCreateTextureObject(&texImage, &texRes, &texDescr, NULL));

  return error;
}

extern "C" cudaError_t CUDA_FreeArray() { return cudaFreeArray(a_Src); }
