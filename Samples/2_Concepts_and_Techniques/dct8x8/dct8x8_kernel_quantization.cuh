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
* \file dct8x8_kernel_quantization.cu
* \brief Contains unoptimized quantization routines. Device code.
*
* This code implements CUDA versions of quantization of Discrete Cosine
* Transform coefficients with 8x8 blocks for float and short arrays.
*/

#pragma once
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "Common.h"

/**
*  JPEG quality=0_of_12 quantization matrix
*/
__constant__ short Q[] = {
  32,  33,  51,  81,  66,  39,  34,  17,
  33,  36,  48,  47,  28,  23,  12,  12,
  51,  48,  47,  28,  23,  12,  12,  12,
  81,  47,  28,  23,  12,  12,  12,  12,
  66,  28,  23,  12,  12,  12,  12,  12,
  39,  23,  12,  12,  12,  12,  12,  12,
  34,  12,  12,  12,  12,  12,  12,  12,
  17,  12,  12,  12,  12,  12,  12,  12
};

/**
**************************************************************************
*  Performs in-place quantization of given DCT coefficients plane using
*  predefined quantization matrices (for floats plane). Unoptimized.
*
* \param SrcDst         [IN/OUT] - DCT coefficients plane
* \param Stride         [IN] - Stride of SrcDst
*
* \return None
*/
__global__ void CUDAkernelQuantizationFloat(float *SrcDst, int Stride) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index (current coefficient)
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // copy current coefficient to the local variable
  float curCoef =
      SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx)];
  float curQuant = (float)Q[ty * BLOCK_SIZE + tx];

  // quantize the current coefficient
  float quantized = roundf(curCoef / curQuant);
  curCoef = quantized * curQuant;

  // copy quantized coefficient back to the DCT-plane
  SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx)] = curCoef;
}

/**
**************************************************************************
*  Performs in-place quantization of given DCT coefficients plane using
*  predefined quantization matrices (for shorts plane). Unoptimized.
*
* \param SrcDst         [IN/OUT] - DCT coefficients plane
* \param Stride         [IN] - Stride of SrcDst
*
* \return None
*/
__global__ void CUDAkernelQuantizationShort(short *SrcDst, int Stride) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index (current coefficient)
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // copy current coefficient to the local variable
  short curCoef =
      SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx)];
  short curQuant = Q[ty * BLOCK_SIZE + tx];

  // quantize the current coefficient
  if (curCoef < 0) {
    curCoef = -curCoef;
    curCoef += curQuant >> 1;
    curCoef /= curQuant;
    curCoef = -curCoef;
  } else {
    curCoef += curQuant >> 1;
    curCoef /= curQuant;
  }

  cg::sync(cta);

  curCoef = curCoef * curQuant;

  // copy quantized coefficient back to the DCT-plane
  SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx)] = curCoef;
}
