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

#ifndef _SIMPLETEXTURE3D_KERNEL_CU_
#define _SIMPLETEXTURE3D_KERNEL_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>
#include <helper_math.h>

typedef unsigned int uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;
cudaTextureObject_t tex;  // 3D texture

__global__ void d_render(uint *d_output, uint imageW, uint imageH, float w,
                         cudaTextureObject_t texObj) {
  uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

  float u = x / (float)imageW;
  float v = y / (float)imageH;
  // read from 3D texture
  float voxel = tex3D<float>(texObj, u, v, w);

  if ((x < imageW) && (y < imageH)) {
    // write output color
    uint i = __umul24(y, imageW) + x;
    d_output[i] = voxel * 255;
  }
}

extern "C" void setTextureFilterMode(bool bLinearFilter) {
  if (tex) {
    checkCudaErrors(cudaDestroyTextureObject(tex));
  }
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = d_volumeArray;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = true;
  texDescr.filterMode =
      bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
  ;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.addressMode[1] = cudaAddressModeWrap;
  texDescr.addressMode[2] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeNormalizedFloat;

  checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
}

extern "C" void initCuda(const uchar *h_volume, cudaExtent volumeSize) {
  // create 3D array
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
  checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

  // copy data to 3D array
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr =
      make_cudaPitchedPtr((void *)h_volume, volumeSize.width * sizeof(uchar),
                          volumeSize.width, volumeSize.height);
  copyParams.dstArray = d_volumeArray;
  copyParams.extent = volumeSize;
  copyParams.kind = cudaMemcpyHostToDevice;
  checkCudaErrors(cudaMemcpy3D(&copyParams));

  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = d_volumeArray;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  // access with normalized texture coordinates
  texDescr.normalizedCoords = true;
  // linear interpolation
  texDescr.filterMode = cudaFilterModeLinear;
  // wrap texture coordinates
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.addressMode[1] = cudaAddressModeWrap;
  texDescr.addressMode[2] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeNormalizedFloat;

  checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
}

extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output,
                              uint imageW, uint imageH, float w) {
  d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, w, tex);
}

void cleanupCuda() {
  if (tex) {
    checkCudaErrors(cudaDestroyTextureObject(tex));
  }
  if (d_volumeArray) {
    checkCudaErrors(cudaFreeArray(d_volumeArray));
  }
}

#endif  // #ifndef _SIMPLETEXTURE3D_KERNEL_CU_
