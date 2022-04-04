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

/**
**************************************************************************
* \file dct8x8_kernel2.cu
* \brief Contains 2nd kernel implementations of DCT and IDCT routines, used in
*        JPEG internal data processing. Optimized device code.
*
* This code implements traditional approach to forward and inverse Discrete
* Cosine Transform to blocks of image pixels (of 8x8 size), as in JPEG standard.
* The data processing is done using floating point representation.
* The routine that performs quantization of coefficients can be found in
* dct8x8_kernel_quantization.cu file.
*/

#pragma once

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include "Common.h"

// Used in forward and inverse DCT
#define C_a 1.387039845322148f  //!< a = (2^0.5) * cos(    pi / 16);
#define C_b 1.306562964876377f  //!< b = (2^0.5) * cos(    pi /  8);
#define C_c 1.175875602419359f  //!< c = (2^0.5) * cos(3 * pi / 16);
#define C_d 0.785694958387102f  //!< d = (2^0.5) * cos(5 * pi / 16);
#define C_e 0.541196100146197f  //!< e = (2^0.5) * cos(3 * pi /  8);
#define C_f 0.275899379282943f  //!< f = (2^0.5) * cos(7 * pi / 16);

/**
*  Normalization constant that is used in forward and inverse DCT
*/
#define C_norm 0.3535533905932737f  // 1 / (8^0.5)

/**
*  Width of data block (2nd kernel)
*/
#define KER2_BLOCK_WIDTH 32

/**
*  Height of data block (2nd kernel)
*/
#define KER2_BLOCK_HEIGHT 16

/**
*  LOG2 of width of data block (2nd kernel)
*/
#define KER2_BW_LOG2 5

/**
*  LOG2 of height of data block (2nd kernel)
*/
#define KER2_BH_LOG2 4

/**
*  Stride of shared memory buffer (2nd kernel)
*/
#define KER2_SMEMBLOCK_STRIDE (KER2_BLOCK_WIDTH + 1)

/**
**************************************************************************
*  Performs in-place DCT of vector of 8 elements.
*
* \param Vect0          [IN/OUT] - Pointer to the first element of vector
* \param Step           [IN/OUT] - Value to add to ptr to access other elements
*
* \return None
*/
__device__ void CUDAsubroutineInplaceDCTvector(float *Vect0, int Step) {
  float *Vect1 = Vect0 + Step;
  float *Vect2 = Vect1 + Step;
  float *Vect3 = Vect2 + Step;
  float *Vect4 = Vect3 + Step;
  float *Vect5 = Vect4 + Step;
  float *Vect6 = Vect5 + Step;
  float *Vect7 = Vect6 + Step;

  float X07P = (*Vect0) + (*Vect7);
  float X16P = (*Vect1) + (*Vect6);
  float X25P = (*Vect2) + (*Vect5);
  float X34P = (*Vect3) + (*Vect4);

  float X07M = (*Vect0) - (*Vect7);
  float X61M = (*Vect6) - (*Vect1);
  float X25M = (*Vect2) - (*Vect5);
  float X43M = (*Vect4) - (*Vect3);

  float X07P34PP = X07P + X34P;
  float X07P34PM = X07P - X34P;
  float X16P25PP = X16P + X25P;
  float X16P25PM = X16P - X25P;

  (*Vect0) = C_norm * (X07P34PP + X16P25PP);
  (*Vect2) = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
  (*Vect4) = C_norm * (X07P34PP - X16P25PP);
  (*Vect6) = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

  (*Vect1) = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
  (*Vect3) = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
  (*Vect5) = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
  (*Vect7) = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

/**
**************************************************************************
*  Performs in-place IDCT of vector of 8 elements.
*
* \param Vect0          [IN/OUT] - Pointer to the first element of vector
* \param Step           [IN/OUT] - Value to add to ptr to access other elements
*
* \return None
*/
__device__ void CUDAsubroutineInplaceIDCTvector(float *Vect0, int Step) {
  float *Vect1 = Vect0 + Step;
  float *Vect2 = Vect1 + Step;
  float *Vect3 = Vect2 + Step;
  float *Vect4 = Vect3 + Step;
  float *Vect5 = Vect4 + Step;
  float *Vect6 = Vect5 + Step;
  float *Vect7 = Vect6 + Step;

  float Y04P = (*Vect0) + (*Vect4);
  float Y2b6eP = C_b * (*Vect2) + C_e * (*Vect6);

  float Y04P2b6ePP = Y04P + Y2b6eP;
  float Y04P2b6ePM = Y04P - Y2b6eP;
  float Y7f1aP3c5dPP =
      C_f * (*Vect7) + C_a * (*Vect1) + C_c * (*Vect3) + C_d * (*Vect5);
  float Y7a1fM3d5cMP =
      C_a * (*Vect7) - C_f * (*Vect1) + C_d * (*Vect3) - C_c * (*Vect5);

  float Y04M = (*Vect0) - (*Vect4);
  float Y2e6bM = C_e * (*Vect2) - C_b * (*Vect6);

  float Y04M2e6bMP = Y04M + Y2e6bM;
  float Y04M2e6bMM = Y04M - Y2e6bM;
  float Y1c7dM3f5aPM =
      C_c * (*Vect1) - C_d * (*Vect7) - C_f * (*Vect3) - C_a * (*Vect5);
  float Y1d7cP3a5fMM =
      C_d * (*Vect1) + C_c * (*Vect7) - C_a * (*Vect3) + C_f * (*Vect5);

  (*Vect0) = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
  (*Vect7) = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
  (*Vect4) = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
  (*Vect3) = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

  (*Vect1) = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
  (*Vect5) = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
  (*Vect2) = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
  (*Vect6) = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

/**
**************************************************************************
*  Performs 8x8 block-wise Forward Discrete Cosine Transform of the given
*  image plane and outputs result to the array of coefficients. 2nd
*implementation.
*  This kernel is designed to process image by blocks of blocks8x8 that
*  utilizes maximum warps capacity, assuming that it is enough of 8 threads
*  per block8x8.
*
* \param SrcDst                     [OUT] - Coefficients plane
* \param ImgStride                  [IN] - Stride of SrcDst
*
* \return None
*/

__global__ void CUDAkernel2DCT(float *dst, float *src, int ImgStride) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  __shared__ float block[KER2_BLOCK_HEIGHT * KER2_SMEMBLOCK_STRIDE];

  int OffsThreadInRow = threadIdx.y * BLOCK_SIZE + threadIdx.x;
  int OffsThreadInCol = threadIdx.z * BLOCK_SIZE;
  src += FMUL(blockIdx.y * KER2_BLOCK_HEIGHT + OffsThreadInCol, ImgStride) +
         blockIdx.x * KER2_BLOCK_WIDTH + OffsThreadInRow;
  dst += FMUL(blockIdx.y * KER2_BLOCK_HEIGHT + OffsThreadInCol, ImgStride) +
         blockIdx.x * KER2_BLOCK_WIDTH + OffsThreadInRow;
  float *bl_ptr =
      block + OffsThreadInCol * KER2_SMEMBLOCK_STRIDE + OffsThreadInRow;

#pragma unroll

  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
    bl_ptr[i * KER2_SMEMBLOCK_STRIDE] = src[i * ImgStride];

  cg::sync(cta);
  // process rows
  CUDAsubroutineInplaceDCTvector(
      block + (OffsThreadInCol + threadIdx.x) * KER2_SMEMBLOCK_STRIDE +
          OffsThreadInRow - threadIdx.x,
      1);

  cg::sync(cta);
  // process columns
  CUDAsubroutineInplaceDCTvector(bl_ptr, KER2_SMEMBLOCK_STRIDE);

  cg::sync(cta);
  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
    dst[i * ImgStride] = bl_ptr[i * KER2_SMEMBLOCK_STRIDE];
}

/**
**************************************************************************
*  Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given
*  coefficients plane and outputs result to the image. 2nd implementation.
*  This kernel is designed to process image by blocks of blocks8x8 that
*  utilizes maximum warps capacity, assuming that it is enough of 8 threads
*  per block8x8.
*
* \param SrcDst                     [OUT] - Coefficients plane
* \param ImgStride                  [IN] - Stride of SrcDst
*
* \return None
*/

__global__ void CUDAkernel2IDCT(float *dst, float *src, int ImgStride) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  __shared__ float block[KER2_BLOCK_HEIGHT * KER2_SMEMBLOCK_STRIDE];

  int OffsThreadInRow = threadIdx.y * BLOCK_SIZE + threadIdx.x;
  int OffsThreadInCol = threadIdx.z * BLOCK_SIZE;
  src += FMUL(blockIdx.y * KER2_BLOCK_HEIGHT + OffsThreadInCol, ImgStride) +
         blockIdx.x * KER2_BLOCK_WIDTH + OffsThreadInRow;
  dst += FMUL(blockIdx.y * KER2_BLOCK_HEIGHT + OffsThreadInCol, ImgStride) +
         blockIdx.x * KER2_BLOCK_WIDTH + OffsThreadInRow;
  float *bl_ptr =
      block + OffsThreadInCol * KER2_SMEMBLOCK_STRIDE + OffsThreadInRow;

#pragma unroll

  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
    bl_ptr[i * KER2_SMEMBLOCK_STRIDE] = src[i * ImgStride];

  cg::sync(cta);
  // process rows
  CUDAsubroutineInplaceIDCTvector(
      block + (OffsThreadInCol + threadIdx.x) * KER2_SMEMBLOCK_STRIDE +
          OffsThreadInRow - threadIdx.x,
      1);

  cg::sync(cta);
  // process columns
  CUDAsubroutineInplaceIDCTvector(bl_ptr, KER2_SMEMBLOCK_STRIDE);

  cg::sync(cta);

  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
    dst[i * ImgStride] = bl_ptr[i * KER2_SMEMBLOCK_STRIDE];
}
