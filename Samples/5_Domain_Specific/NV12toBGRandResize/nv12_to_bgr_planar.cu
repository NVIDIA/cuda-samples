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


// Implements NV12 to BGR batch conversion

#include <cuda.h>
#include <cuda_runtime.h>

#include "resize_convert.h"

#define CONV_THREADS_X 64
#define CONV_THREADS_Y 10

__forceinline__ __device__ static float clampF(float x, float lower,
                                               float upper) {
  return x < lower ? lower : (x > upper ? upper : x);
}

__global__ static void nv12ToBGRplanarBatchKernel(const uint8_t *pNv12,
                                                  int nNv12Pitch, float *pBgr,
                                                  int nRgbPitch, int nWidth,
                                                  int nHeight, int nBatchSize) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if ((x << 2) + 1 > nWidth || (y << 1) + 1 > nHeight) return;

  const uint8_t *__restrict__ pSrc = pNv12;

  for (int i = blockIdx.z; i < nBatchSize; i += gridDim.z) {
    pSrc = pNv12 + i * ((nHeight * nNv12Pitch * 3) >> 1) + (x << 2) +
           (y << 1) * nNv12Pitch;
    uchar4 luma2x01, luma2x23, uv2;
    *(uint32_t *)&luma2x01 = *(uint32_t *)pSrc;
    *(uint32_t *)&luma2x23 = *(uint32_t *)(pSrc + nNv12Pitch);
    *(uint32_t *)&uv2 = *(uint32_t *)(pSrc + (nHeight - y) * nNv12Pitch);

    float *pDstBlock = (pBgr + i * ((nHeight * nRgbPitch * 3) >> 2) +
                        ((blockIdx.x * blockDim.x) << 2) +
                        ((blockIdx.y * blockDim.y) << 1) * (nRgbPitch >> 2));

    float2 add1;
    float2 add2;
    float2 add3;
    float2 add00, add01, add02, add03;
    float2 d, e;

    add00.x = 1.1644f * luma2x01.x;
    add01.x = 1.1644f * luma2x01.y;
    add00.y = 1.1644f * luma2x01.z;
    add01.y = 1.1644f * luma2x01.w;

    add02.x = 1.1644f * luma2x23.x;
    add03.x = 1.1644f * luma2x23.y;
    add02.y = 1.1644f * luma2x23.z;
    add03.y = 1.1644f * luma2x23.w;

    d.x = uv2.x - 128.0f;
    e.x = uv2.y - 128.0f;
    d.y = uv2.z - 128.0f;
    e.y = uv2.w - 128.0f;

    add1.x = 2.0172f * d.x;
    add1.y = 2.0172f * d.y;

    add2.x = (-0.3918f) * d.x + (-0.8130f) * e.x;
    add2.y = (-0.3918f) * d.y + (-0.8130f) * e.y;

    add3.x = 1.5960f * e.x;
    add3.y = 1.5960f * e.y;

    int rowStride = (threadIdx.y << 1) * (nRgbPitch >> 2);
    int nextRowStride = ((threadIdx.y << 1) + 1) * (nRgbPitch >> 2);
    // B
    *((float4 *)&pDstBlock[rowStride + (threadIdx.x << 2)]) =
        make_float4(clampF(add00.x + add1.x, 0.0f, 255.0f),
                    clampF(add01.x + add1.x, 0.0f, 255.0f),
                    clampF(add00.y + add1.y, 0.0f, 255.0f),
                    clampF(add01.y + add1.y, 0.0f, 255.0f));
    *((float4 *)&pDstBlock[nextRowStride + (threadIdx.x << 2)]) =
        make_float4(clampF(add02.x + add1.x, 0.0f, 255.0f),
                    clampF(add03.x + add1.x, 0.0f, 255.0f),
                    clampF(add02.y + add1.y, 0.0f, 255.0f),
                    clampF(add03.y + add1.y, 0.0f, 255.0f));

    int planeStride = nHeight * nRgbPitch >> 2;
    // G
    *((float4 *)&pDstBlock[planeStride + rowStride + (threadIdx.x << 2)]) =
        make_float4(clampF(add00.x + add2.x, 0.0f, 255.0f),
                    clampF(add01.x + add2.x, 0.0f, 255.0f),
                    clampF(add00.y + add2.y, 0.0f, 255.0f),
                    clampF(add01.y + add2.y, 0.0f, 255.0f));
    *((float4 *)&pDstBlock[planeStride + nextRowStride + (threadIdx.x << 2)]) =
        make_float4(clampF(add02.x + add2.x, 0.0f, 255.0f),
                    clampF(add03.x + add2.x, 0.0f, 255.0f),
                    clampF(add02.y + add2.y, 0.0f, 255.0f),
                    clampF(add03.y + add2.y, 0.0f, 255.0f));

    // R
    *((float4
           *)&pDstBlock[(planeStride << 1) + rowStride + (threadIdx.x << 2)]) =
        make_float4(clampF(add00.x + add3.x, 0.0f, 255.0f),
                    clampF(add01.x + add3.x, 0.0f, 255.0f),
                    clampF(add00.y + add3.y, 0.0f, 255.0f),
                    clampF(add01.y + add3.y, 0.0f, 255.0f));
    *((float4 *)&pDstBlock[(planeStride << 1) + nextRowStride +
                           (threadIdx.x << 2)]) =
        make_float4(clampF(add02.x + add3.x, 0.0f, 255.0f),
                    clampF(add03.x + add3.x, 0.0f, 255.0f),
                    clampF(add02.y + add3.y, 0.0f, 255.0f),
                    clampF(add03.y + add3.y, 0.0f, 255.0f));
  }
}

void nv12ToBGRplanarBatch(uint8_t *pNv12, int nNv12Pitch, float *pBgr,
                          int nRgbPitch, int nWidth, int nHeight,
                          int nBatchSize, cudaStream_t stream) {
  dim3 threads(CONV_THREADS_X, CONV_THREADS_Y);

  size_t blockDimZ = nBatchSize;

  // Restricting blocks in Z-dim till 32 to not launch too many blocks
  blockDimZ = (blockDimZ > 32) ? 32 : blockDimZ;

  dim3 blocks((nWidth / 4 - 1) / threads.x + 1,
              (nHeight / 2 - 1) / threads.y + 1, blockDimZ);
  nv12ToBGRplanarBatchKernel<<<blocks, threads, 0, stream>>>(
      pNv12, nNv12Pitch, pBgr, nRgbPitch, nWidth, nHeight, nBatchSize);
}
