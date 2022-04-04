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

#ifndef IMAGE_DENOISING_H
#define IMAGE_DENOISING_H

typedef unsigned int TColor;

////////////////////////////////////////////////////////////////////////////////
// Filter configuration
////////////////////////////////////////////////////////////////////////////////
#define KNN_WINDOW_RADIUS 3
#define NLM_WINDOW_RADIUS 3
#define NLM_BLOCK_RADIUS 3
#define KNN_WINDOW_AREA \
  ((2 * KNN_WINDOW_RADIUS + 1) * (2 * KNN_WINDOW_RADIUS + 1))
#define NLM_WINDOW_AREA \
  ((2 * NLM_WINDOW_RADIUS + 1) * (2 * NLM_WINDOW_RADIUS + 1))
#define INV_KNN_WINDOW_AREA (1.0f / (float)KNN_WINDOW_AREA)
#define INV_NLM_WINDOW_AREA (1.0f / (float)NLM_WINDOW_AREA)

#define KNN_WEIGHT_THRESHOLD 0.02f
#define KNN_LERP_THRESHOLD 0.79f
#define NLM_WEIGHT_THRESHOLD 0.10f
#define NLM_LERP_THRESHOLD 0.10f

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

#ifndef MAX
#define MAX(a, b) ((a < b) ? b : a)
#endif
#ifndef MIN
#define MIN(a, b) ((a < b) ? a : b)
#endif

// functions to load images
extern "C" void LoadBMPFile(uchar4 **dst, int *width, int *height,
                            const char *name);

// CUDA wrapper functions for allocation/freeing texture arrays
extern "C" cudaTextureObject_t texImage;

extern "C" cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH);
extern "C" cudaError_t CUDA_FreeArray();

// CUDA kernel functions
extern "C" void cuda_Copy(TColor *d_dst, int imageW, int imageH,
                          cudaTextureObject_t texImage);
extern "C" void cuda_KNN(TColor *d_dst, int imageW, int imageH, float Noise,
                         float lerpC, cudaTextureObject_t texImage);
extern "C" void cuda_KNNdiag(TColor *d_dst, int imageW, int imageH, float Noise,
                             float lerpC, cudaTextureObject_t texImage);
extern "C" void cuda_NLM(TColor *d_dst, int imageW, int imageH, float Noise,
                         float lerpC, cudaTextureObject_t texImage);
extern "C" void cuda_NLMdiag(TColor *d_dst, int imageW, int imageH, float Noise,
                             float lerpC, cudaTextureObject_t texImage);

extern "C" void cuda_NLM2(TColor *d_dst, int imageW, int imageH, float Noise,
                          float LerpC, cudaTextureObject_t texImage);
extern "C" void cuda_NLM2diag(TColor *d_dst, int imageW, int imageH,
                              float Noise, float LerpC,
                              cudaTextureObject_t texImage);

#endif
