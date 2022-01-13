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

#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief downscale image
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
__global__ void DownscaleKernel(int width, int height, int stride, float *out,
                                cudaTextureObject_t texFine) {
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  if (ix >= width || iy >= height) {
    return;
  }

  float dx = 1.0f / (float)width;
  float dy = 1.0f / (float)height;

  float x = ((float)ix + 0.5f) * dx;
  float y = ((float)iy + 0.5f) * dy;

  out[ix + iy * stride] = 0.25f * (tex2D<float>(texFine, x - dx * 0.25f, y) +
                                   tex2D<float>(texFine, x + dx * 0.25f, y) +
                                   tex2D<float>(texFine, x, y - dy * 0.25f) +
                                   tex2D<float>(texFine, x, y + dy * 0.25f));
}

///////////////////////////////////////////////////////////////////////////////
/// \brief downscale image
///
/// \param[in]  src     image to downscale
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
static void Downscale(const float *src, int width, int height, int stride,
                      int newWidth, int newHeight, int newStride, float *out) {
  dim3 threads(32, 8);
  dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

  cudaTextureObject_t texFine;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypePitch2D;
  texRes.res.pitch2D.devPtr = (void *)src;
  texRes.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  texRes.res.pitch2D.width = width;
  texRes.res.pitch2D.height = height;
  texRes.res.pitch2D.pitchInBytes = stride * sizeof(float);

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = true;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeMirror;
  texDescr.addressMode[1] = cudaAddressModeMirror;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(cudaCreateTextureObject(&texFine, &texRes, &texDescr, NULL));

  DownscaleKernel<<<blocks, threads>>>(newWidth, newHeight, newStride, out,
                                       texFine);
}
