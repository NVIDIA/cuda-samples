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

#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_math.h>

// includes, cuda
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#include "bicubicTexture_kernel.cuh"

cudaArray *d_imageArray = 0;

extern "C" void initTexture(int imageWidth, int imageHeight, uchar *h_data) {
  // allocate array and copy image data
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
  checkCudaErrors(
      cudaMallocArray(&d_imageArray, &channelDesc, imageWidth, imageHeight));
  checkCudaErrors(cudaMemcpy2DToArray(
      d_imageArray, 0, 0, h_data, imageWidth * sizeof(uchar),
      imageWidth * sizeof(uchar), imageHeight, cudaMemcpyHostToDevice));
  free(h_data);

  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = d_imageArray;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeClamp;
  texDescr.addressMode[1] = cudaAddressModeClamp;
  texDescr.readMode = cudaReadModeNormalizedFloat;

  checkCudaErrors(
      cudaCreateTextureObject(&texObjLinear, &texRes, &texDescr, NULL));

  memset(&texDescr, 0, sizeof(cudaTextureDesc));
  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeClamp;
  texDescr.addressMode[1] = cudaAddressModeClamp;
  texDescr.readMode = cudaReadModeNormalizedFloat;

  checkCudaErrors(
      cudaCreateTextureObject(&texObjPoint, &texRes, &texDescr, NULL));
}

extern "C" void freeTexture() {
  checkCudaErrors(cudaDestroyTextureObject(texObjPoint));
  checkCudaErrors(cudaDestroyTextureObject(texObjLinear));
  checkCudaErrors(cudaFreeArray(d_imageArray));
}

// render image using CUDA
extern "C" void render(int width, int height, float tx, float ty, float scale,
                       float cx, float cy, dim3 blockSize, dim3 gridSize,
                       int filter_mode, uchar4 *output) {
  // call CUDA kernel, writing results to PBO memory
  switch (filter_mode) {
    case MODE_NEAREST:
      d_render<<<gridSize, blockSize>>>(output, width, height, tx, ty, scale,
                                        cx, cy, texObjPoint);
      break;

    case MODE_BILINEAR:
      d_render<<<gridSize, blockSize>>>(output, width, height, tx, ty, scale,
                                        cx, cy, texObjLinear);
      break;

    case MODE_BICUBIC:
      d_renderBicubic<<<gridSize, blockSize>>>(output, width, height, tx, ty,
                                               scale, cx, cy, texObjPoint);
      break;

    case MODE_FAST_BICUBIC:
      d_renderFastBicubic<<<gridSize, blockSize>>>(
          output, width, height, tx, ty, scale, cx, cy, texObjLinear);
      break;

    case MODE_CATROM:
      d_renderCatRom<<<gridSize, blockSize>>>(output, width, height, tx, ty,
                                              scale, cx, cy, texObjPoint);
      break;
  }

  getLastCudaError("kernel failed");
}

#endif
