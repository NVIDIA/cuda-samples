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
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <helper_string.h>

#include "SobelFilter_kernels.h"

// Texture object for reading image
cudaTextureObject_t texObject;
extern __shared__ unsigned char LocalBlock[];
static cudaArray *array = NULL;

#define RADIUS 1

#ifdef FIXED_BLOCKWIDTH
#define BlockWidth 80
#define SharedPitch 384
#endif

// This will output the proper CUDA error strings in the event that a CUDA host
// call returns an error
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
            (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__device__ unsigned char ComputeSobel(unsigned char ul,  // upper left
                                      unsigned char um,  // upper middle
                                      unsigned char ur,  // upper right
                                      unsigned char ml,  // middle left
                                      unsigned char mm,  // middle (unused)
                                      unsigned char mr,  // middle right
                                      unsigned char ll,  // lower left
                                      unsigned char lm,  // lower middle
                                      unsigned char lr,  // lower right
                                      float fScale) {
  short Horz = ur + 2 * mr + lr - ul - 2 * ml - ll;
  short Vert = ul + 2 * um + ur - ll - 2 * lm - lr;
  short Sum = (short)(fScale * (abs((int)Horz) + abs((int)Vert)));

  if (Sum < 0) {
    return 0;
  } else if (Sum > 0xff) {
    return 0xff;
  }

  return (unsigned char)Sum;
}

__global__ void SobelShared(uchar4 *pSobelOriginal, unsigned short SobelPitch,
#ifndef FIXED_BLOCKWIDTH
                            short BlockWidth, short SharedPitch,
#endif
                            short w, short h, float fScale,
                            cudaTextureObject_t tex) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  short u = 4 * blockIdx.x * BlockWidth;
  short v = blockIdx.y * blockDim.y + threadIdx.y;
  short ib;

  int SharedIdx = threadIdx.y * SharedPitch;

  for (ib = threadIdx.x; ib < BlockWidth + 2 * RADIUS; ib += blockDim.x) {
    LocalBlock[SharedIdx + 4 * ib + 0] = tex2D<unsigned char>(
        tex, (float)(u + 4 * ib - RADIUS + 0), (float)(v - RADIUS));
    LocalBlock[SharedIdx + 4 * ib + 1] = tex2D<unsigned char>(
        tex, (float)(u + 4 * ib - RADIUS + 1), (float)(v - RADIUS));
    LocalBlock[SharedIdx + 4 * ib + 2] = tex2D<unsigned char>(
        tex, (float)(u + 4 * ib - RADIUS + 2), (float)(v - RADIUS));
    LocalBlock[SharedIdx + 4 * ib + 3] = tex2D<unsigned char>(
        tex, (float)(u + 4 * ib - RADIUS + 3), (float)(v - RADIUS));
  }

  if (threadIdx.y < RADIUS * 2) {
    //
    // copy trailing RADIUS*2 rows of pixels into shared
    //
    SharedIdx = (blockDim.y + threadIdx.y) * SharedPitch;

    for (ib = threadIdx.x; ib < BlockWidth + 2 * RADIUS; ib += blockDim.x) {
      LocalBlock[SharedIdx + 4 * ib + 0] =
          tex2D<unsigned char>(tex, (float)(u + 4 * ib - RADIUS + 0),
                               (float)(v + blockDim.y - RADIUS));
      LocalBlock[SharedIdx + 4 * ib + 1] =
          tex2D<unsigned char>(tex, (float)(u + 4 * ib - RADIUS + 1),
                               (float)(v + blockDim.y - RADIUS));
      LocalBlock[SharedIdx + 4 * ib + 2] =
          tex2D<unsigned char>(tex, (float)(u + 4 * ib - RADIUS + 2),
                               (float)(v + blockDim.y - RADIUS));
      LocalBlock[SharedIdx + 4 * ib + 3] =
          tex2D<unsigned char>(tex, (float)(u + 4 * ib - RADIUS + 3),
                               (float)(v + blockDim.y - RADIUS));
    }
  }

  cg::sync(cta);

  u >>= 2;  // index as uchar4 from here
  uchar4 *pSobel = (uchar4 *)(((char *)pSobelOriginal) + v * SobelPitch);
  SharedIdx = threadIdx.y * SharedPitch;

  for (ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x) {
    unsigned char pix00 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 0];
    unsigned char pix01 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 1];
    unsigned char pix02 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 2];
    unsigned char pix10 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 0];
    unsigned char pix11 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 1];
    unsigned char pix12 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 2];
    unsigned char pix20 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 0];
    unsigned char pix21 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 1];
    unsigned char pix22 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 2];

    uchar4 out;

    out.x = ComputeSobel(pix00, pix01, pix02, pix10, pix11, pix12, pix20, pix21,
                         pix22, fScale);

    pix00 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 3];
    pix10 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 3];
    pix20 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 3];
    out.y = ComputeSobel(pix01, pix02, pix00, pix11, pix12, pix10, pix21, pix22,
                         pix20, fScale);

    pix01 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 4];
    pix11 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 4];
    pix21 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 4];
    out.z = ComputeSobel(pix02, pix00, pix01, pix12, pix10, pix11, pix22, pix20,
                         pix21, fScale);

    pix02 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 5];
    pix12 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 5];
    pix22 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 5];
    out.w = ComputeSobel(pix00, pix01, pix02, pix10, pix11, pix12, pix20, pix21,
                         pix22, fScale);

    if (u + ib < w / 4 && v < h) {
      pSobel[u + ib] = out;
    }
  }

  cg::sync(cta);
}

__global__ void SobelCopyImage(Pixel *pSobelOriginal, unsigned int Pitch, int w,
                               int h, float fscale, cudaTextureObject_t tex) {
  unsigned char *pSobel =
      (unsigned char *)(((char *)pSobelOriginal) + blockIdx.x * Pitch);

  for (int i = threadIdx.x; i < w; i += blockDim.x) {
    pSobel[i] = min(
        max((tex2D<unsigned char>(tex, (float)i, (float)blockIdx.x) * fscale),
            0.f),
        255.f);
  }
}

__global__ void SobelTex(Pixel *pSobelOriginal, unsigned int Pitch, int w,
                         int h, float fScale, cudaTextureObject_t tex) {
  unsigned char *pSobel =
      (unsigned char *)(((char *)pSobelOriginal) + blockIdx.x * Pitch);

  for (int i = threadIdx.x; i < w; i += blockDim.x) {
    unsigned char pix00 =
        tex2D<unsigned char>(tex, (float)i - 1, (float)blockIdx.x - 1);
    unsigned char pix01 =
        tex2D<unsigned char>(tex, (float)i + 0, (float)blockIdx.x - 1);
    unsigned char pix02 =
        tex2D<unsigned char>(tex, (float)i + 1, (float)blockIdx.x - 1);
    unsigned char pix10 =
        tex2D<unsigned char>(tex, (float)i - 1, (float)blockIdx.x + 0);
    unsigned char pix11 =
        tex2D<unsigned char>(tex, (float)i + 0, (float)blockIdx.x + 0);
    unsigned char pix12 =
        tex2D<unsigned char>(tex, (float)i + 1, (float)blockIdx.x + 0);
    unsigned char pix20 =
        tex2D<unsigned char>(tex, (float)i - 1, (float)blockIdx.x + 1);
    unsigned char pix21 =
        tex2D<unsigned char>(tex, (float)i + 0, (float)blockIdx.x + 1);
    unsigned char pix22 =
        tex2D<unsigned char>(tex, (float)i + 1, (float)blockIdx.x + 1);
    pSobel[i] = ComputeSobel(pix00, pix01, pix02, pix10, pix11, pix12, pix20,
                             pix21, pix22, fScale);
  }
}

extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp) {
  cudaChannelFormatDesc desc;

  if (Bpp == 1) {
    desc = cudaCreateChannelDesc<unsigned char>();
  } else {
    desc = cudaCreateChannelDesc<uchar4>();
  }

  checkCudaErrors(cudaMallocArray(&array, &desc, iw, ih));
  checkCudaErrors(cudaMemcpy2DToArray(
      array, 0, 0, data, iw * Bpp * sizeof(Pixel), iw * Bpp * sizeof(Pixel), ih,
      cudaMemcpyHostToDevice));

  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = array;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(
      cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));
}

extern "C" void deleteTexture(void) {
  checkCudaErrors(cudaFreeArray(array));
  checkCudaErrors(cudaDestroyTextureObject(texObject));
}

// Wrapper for the __global__ call that sets up the texture and threads
extern "C" void sobelFilter(Pixel *odata, int iw, int ih,
                            enum SobelDisplayMode mode, float fScale) {
  switch (mode) {
    case SOBELDISPLAY_IMAGE:
      SobelCopyImage<<<ih, 384>>>(odata, iw, iw, ih, fScale, texObject);
      break;

    case SOBELDISPLAY_SOBELTEX:
      SobelTex<<<ih, 384>>>(odata, iw, iw, ih, fScale, texObject);
      break;

    case SOBELDISPLAY_SOBELSHARED: {
      dim3 threads(16, 4);
#ifndef FIXED_BLOCKWIDTH
      int BlockWidth = 80;  // must be divisible by 16 for coalescing
#endif
      dim3 blocks = dim3(iw / (4 * BlockWidth) + (0 != iw % (4 * BlockWidth)),
                         ih / threads.y + (0 != ih % threads.y));
      int SharedPitch = ~0x3f & (4 * (BlockWidth + 2 * RADIUS) + 0x3f);
      int sharedMem = SharedPitch * (threads.y + 2 * RADIUS);

      // for the shared kernel, width must be divisible by 4
      iw &= ~3;

      SobelShared<<<blocks, threads, sharedMem>>>((uchar4 *)odata, iw,
#ifndef FIXED_BLOCKWIDTH
                                                  BlockWidth, SharedPitch,
#endif
                                                  iw, ih, fScale, texObject);
    } break;
  }
}
