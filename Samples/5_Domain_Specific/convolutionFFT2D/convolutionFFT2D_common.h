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

#ifndef CONVOLUTIONFFT2D_COMMON_H
#define CONVOLUTIONFFT2D_COMMON_H

typedef unsigned int uint;

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct {
  float x;
  float y;
} fComplex;
#endif

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
// Round a / b to nearest higher integer value
inline int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }

extern "C" void convolutionClampToBorderCPU(float *h_Result, float *h_Data,
                                            float *h_Kernel, int dataH,
                                            int dataW, int kernelH, int kernelW,
                                            int kernelY, int kernelX);

extern "C" void padKernel(float *d_PaddedKernel, float *d_Kernel, int fftH,
                          int fftW, int kernelH, int kernelW, int kernelY,
                          int kernelX);

extern "C" void padDataClampToBorder(float *d_PaddedData, float *d_Data,
                                     int fftH, int fftW, int dataH, int dataW,
                                     int kernelH, int kernelW, int kernelY,
                                     int kernelX);

extern "C" void modulateAndNormalize(fComplex *d_Dst, fComplex *d_Src, int fftH,
                                     int fftW, int padding);

extern "C" void spPostprocess2D(void *d_Dst, void *d_Src, uint DY, uint DX,
                                uint padding, int dir);

extern "C" void spPreprocess2D(void *d_Dst, void *d_Src, uint DY, uint DX,
                               uint padding, int dir);

extern "C" void spProcess2D(void *d_Data, void *d_Data0, void *d_Kernel0,
                            uint DY, uint DX, int dir);

#endif  // CONVOLUTIONFFT2D_COMMON_H
