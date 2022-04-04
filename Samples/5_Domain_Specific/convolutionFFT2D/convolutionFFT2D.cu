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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include "convolutionFFT2D_common.h"
#include "convolutionFFT2D.cuh"

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
extern "C" void padKernel(float *d_Dst, float *d_Src, int fftH, int fftW,
                          int kernelH, int kernelW, int kernelY, int kernelX) {
  assert(d_Src != d_Dst);
  dim3 threads(32, 8);
  dim3 grid(iDivUp(kernelW, threads.x), iDivUp(kernelH, threads.y));

  SET_FLOAT_BASE;
#if (USE_TEXTURE)
  cudaTextureObject_t texFloat;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = d_Src;
  texRes.res.linear.sizeInBytes = sizeof(float) * kernelH * kernelW;
  texRes.res.linear.desc = cudaCreateChannelDesc<float>();

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(cudaCreateTextureObject(&texFloat, &texRes, &texDescr, NULL));
#endif

  padKernel_kernel<<<grid, threads>>>(d_Dst, d_Src, fftH, fftW, kernelH,
                                      kernelW, kernelY, kernelX
#if (USE_TEXTURE)
                                      ,
                                      texFloat
#endif
                                      );
  getLastCudaError("padKernel_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
  checkCudaErrors(cudaDestroyTextureObject(texFloat));
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataClampToBorder(float *d_Dst, float *d_Src, int fftH,
                                     int fftW, int dataH, int dataW,
                                     int kernelW, int kernelH, int kernelY,
                                     int kernelX) {
  assert(d_Src != d_Dst);
  dim3 threads(32, 8);
  dim3 grid(iDivUp(fftW, threads.x), iDivUp(fftH, threads.y));

#if (USE_TEXTURE)
  cudaTextureObject_t texFloat;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = d_Src;
  texRes.res.linear.sizeInBytes = sizeof(float) * dataH * dataW;
  texRes.res.linear.desc = cudaCreateChannelDesc<float>();

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(cudaCreateTextureObject(&texFloat, &texRes, &texDescr, NULL));
#endif

  padDataClampToBorder_kernel<<<grid, threads>>>(
      d_Dst, d_Src, fftH, fftW, dataH, dataW, kernelH, kernelW, kernelY, kernelX
#if (USE_TEXTURE)
      ,
      texFloat
#endif
      );
  getLastCudaError("padDataClampToBorder_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
  checkCudaErrors(cudaDestroyTextureObject(texFloat));
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
extern "C" void modulateAndNormalize(fComplex *d_Dst, fComplex *d_Src, int fftH,
                                     int fftW, int padding) {
  assert(fftW % 2 == 0);
  const int dataSize = fftH * (fftW / 2 + padding);

  modulateAndNormalize_kernel<<<iDivUp(dataSize, 256), 256>>>(
      d_Dst, d_Src, dataSize, 1.0f / (float)(fftW * fftH));
  getLastCudaError("modulateAndNormalize() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// 2D R2C / C2R post/preprocessing kernels
////////////////////////////////////////////////////////////////////////////////
static const double PI = 3.1415926535897932384626433832795;
static const uint BLOCKDIM = 256;

extern "C" void spPostprocess2D(void *d_Dst, void *d_Src, uint DY, uint DX,
                                uint padding, int dir) {
  assert(d_Src != d_Dst);
  assert(DX % 2 == 0);

#if (POWER_OF_TWO)
  uint log2DX, log2DY;
  uint factorizationRemX = factorRadix2(log2DX, DX);
  uint factorizationRemY = factorRadix2(log2DY, DY);
  assert(factorizationRemX == 1 && factorizationRemY == 1);
#endif

  const uint threadCount = DY * (DX / 2);
  const double phaseBase = dir * PI / (double)DX;

#if (USE_TEXTURE)
  cudaTextureObject_t texComplex;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = d_Src;
  texRes.res.linear.sizeInBytes = sizeof(fComplex) * DY * (DX + padding);
  texRes.res.linear.desc = cudaCreateChannelDesc<fComplex>();

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(
      cudaCreateTextureObject(&texComplex, &texRes, &texDescr, NULL));
#endif

  spPostprocess2D_kernel<<<iDivUp(threadCount, BLOCKDIM), BLOCKDIM>>>(
      (fComplex *)d_Dst, (fComplex *)d_Src, DY, DX, threadCount, padding,
      (float)phaseBase
#if (USE_TEXTURE)
      ,
      texComplex
#endif
      );
  getLastCudaError("spPostprocess2D_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
  checkCudaErrors(cudaDestroyTextureObject(texComplex));
#endif
}

extern "C" void spPreprocess2D(void *d_Dst, void *d_Src, uint DY, uint DX,
                               uint padding, int dir) {
  assert(d_Src != d_Dst);
  assert(DX % 2 == 0);

#if (POWER_OF_TWO)
  uint log2DX, log2DY;
  uint factorizationRemX = factorRadix2(log2DX, DX);
  uint factorizationRemY = factorRadix2(log2DY, DY);
  assert(factorizationRemX == 1 && factorizationRemY == 1);
#endif

  const uint threadCount = DY * (DX / 2);
  const double phaseBase = -dir * PI / (double)DX;

#if (USE_TEXTURE)
  cudaTextureObject_t texComplex;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = d_Src;
  texRes.res.linear.sizeInBytes = sizeof(fComplex) * DY * (DX + padding);
  texRes.res.linear.desc = cudaCreateChannelDesc<fComplex>();

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(
      cudaCreateTextureObject(&texComplex, &texRes, &texDescr, NULL));
#endif
  spPreprocess2D_kernel<<<iDivUp(threadCount, BLOCKDIM), BLOCKDIM>>>(
      (fComplex *)d_Dst, (fComplex *)d_Src, DY, DX, threadCount, padding,
      (float)phaseBase
#if (USE_TEXTURE)
      ,
      texComplex
#endif
      );
  getLastCudaError("spPreprocess2D_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
  checkCudaErrors(cudaDestroyTextureObject(texComplex));
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Combined spPostprocess2D + modulateAndNormalize + spPreprocess2D
////////////////////////////////////////////////////////////////////////////////
extern "C" void spProcess2D(void *d_Dst, void *d_SrcA, void *d_SrcB, uint DY,
                            uint DX, int dir) {
  assert(DY % 2 == 0);

#if (POWER_OF_TWO)
  uint log2DX, log2DY;
  uint factorizationRemX = factorRadix2(log2DX, DX);
  uint factorizationRemY = factorRadix2(log2DY, DY);
  assert(factorizationRemX == 1 && factorizationRemY == 1);
#endif

  const uint threadCount = (DY / 2) * DX;
  const double phaseBase = dir * PI / (double)DX;

#if (USE_TEXTURE)
  cudaTextureObject_t texComplexA, texComplexB;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = d_SrcA;
  texRes.res.linear.sizeInBytes = sizeof(fComplex) * DY * DX;
  texRes.res.linear.desc = cudaCreateChannelDesc<fComplex>();

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(
      cudaCreateTextureObject(&texComplexA, &texRes, &texDescr, NULL));

  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = d_SrcB;
  texRes.res.linear.sizeInBytes = sizeof(fComplex) * DY * DX;
  texRes.res.linear.desc = cudaCreateChannelDesc<fComplex>();

  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(
      cudaCreateTextureObject(&texComplexB, &texRes, &texDescr, NULL));
#endif
  spProcess2D_kernel<<<iDivUp(threadCount, BLOCKDIM), BLOCKDIM>>>(
      (fComplex *)d_Dst, (fComplex *)d_SrcA, (fComplex *)d_SrcB, DY, DX,
      threadCount, (float)phaseBase, 0.5f / (float)(DY * DX)
#if (USE_TEXTURE)
                                         ,
      texComplexA, texComplexB
#endif
      );
  getLastCudaError("spProcess2D_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
  checkCudaErrors(cudaDestroyTextureObject(texComplexA));
  checkCudaErrors(cudaDestroyTextureObject(texComplexB));
#endif
}
