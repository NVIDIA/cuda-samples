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
/// \brief compute image derivatives
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] Ix      x derivative
/// \param[out] Iy      y derivative
/// \param[out] Iz      temporal derivative
///////////////////////////////////////////////////////////////////////////////
__global__ void ComputeDerivativesKernel(int width, int height, int stride,
                                         float *Ix, float *Iy, float *Iz,
                                         cudaTextureObject_t texSource,
                                         cudaTextureObject_t texTarget) {
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  const int pos = ix + iy * stride;

  if (ix >= width || iy >= height) return;

  float dx = 1.0f / (float)width;
  float dy = 1.0f / (float)height;

  float x = ((float)ix + 0.5f) * dx;
  float y = ((float)iy + 0.5f) * dy;

  float t0, t1;
  // x derivative
  t0 = tex2D<float>(texSource, x - 2.0f * dx, y);
  t0 -= tex2D<float>(texSource, x - 1.0f * dx, y) * 8.0f;
  t0 += tex2D<float>(texSource, x + 1.0f * dx, y) * 8.0f;
  t0 -= tex2D<float>(texSource, x + 2.0f * dx, y);
  t0 /= 12.0f;

  t1 = tex2D<float>(texTarget, x - 2.0f * dx, y);
  t1 -= tex2D<float>(texTarget, x - 1.0f * dx, y) * 8.0f;
  t1 += tex2D<float>(texTarget, x + 1.0f * dx, y) * 8.0f;
  t1 -= tex2D<float>(texTarget, x + 2.0f * dx, y);
  t1 /= 12.0f;

  Ix[pos] = (t0 + t1) * 0.5f;

  // t derivative
  Iz[pos] = tex2D<float>(texTarget, x, y) - tex2D<float>(texSource, x, y);

  // y derivative
  t0 = tex2D<float>(texSource, x, y - 2.0f * dy);
  t0 -= tex2D<float>(texSource, x, y - 1.0f * dy) * 8.0f;
  t0 += tex2D<float>(texSource, x, y + 1.0f * dy) * 8.0f;
  t0 -= tex2D<float>(texSource, x, y + 2.0f * dy);
  t0 /= 12.0f;

  t1 = tex2D<float>(texTarget, x, y - 2.0f * dy);
  t1 -= tex2D<float>(texTarget, x, y - 1.0f * dy) * 8.0f;
  t1 += tex2D<float>(texTarget, x, y + 1.0f * dy) * 8.0f;
  t1 -= tex2D<float>(texTarget, x, y + 2.0f * dy);
  t1 /= 12.0f;

  Iy[pos] = (t0 + t1) * 0.5f;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// \param[in]  I0  source image
/// \param[in]  I1  tracked image
/// \param[in]  w   image width
/// \param[in]  h   image height
/// \param[in]  s   image stride
/// \param[out] Ix  x derivative
/// \param[out] Iy  y derivative
/// \param[out] Iz  temporal derivative
///////////////////////////////////////////////////////////////////////////////
static void ComputeDerivatives(const float *I0, const float *I1, int w, int h,
                               int s, float *Ix, float *Iy, float *Iz) {
  dim3 threads(32, 6);
  dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

  cudaTextureObject_t texSource, texTarget;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypePitch2D;
  texRes.res.pitch2D.devPtr = (void *)I0;
  texRes.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  texRes.res.pitch2D.width = w;
  texRes.res.pitch2D.height = h;
  texRes.res.pitch2D.pitchInBytes = s * sizeof(float);

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = true;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeMirror;
  texDescr.addressMode[1] = cudaAddressModeMirror;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(
      cudaCreateTextureObject(&texSource, &texRes, &texDescr, NULL));
  memset(&texRes, 0, sizeof(cudaResourceDesc));
  texRes.resType = cudaResourceTypePitch2D;
  texRes.res.pitch2D.devPtr = (void *)I1;
  texRes.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  texRes.res.pitch2D.width = w;
  texRes.res.pitch2D.height = h;
  texRes.res.pitch2D.pitchInBytes = s * sizeof(float);
  checkCudaErrors(
      cudaCreateTextureObject(&texTarget, &texRes, &texDescr, NULL));

  ComputeDerivativesKernel<<<blocks, threads>>>(w, h, s, Ix, Iy, Iz, texSource,
                                                texTarget);
}
