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

/* Simple kernel computes a Stereo Disparity using CUDA SIMD SAD intrinsics. */

#ifndef _STEREODISPARITY_KERNEL_H_
#define _STEREODISPARITY_KERNEL_H_

#define blockSize_x 32
#define blockSize_y 8

// RAD is the radius of the region of support for the search
#define RAD 8
// STEPS is the number of loads we must perform to initialize the shared memory
// area (see convolution CUDA Sample for example)
#define STEPS 3

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////
// This function applies the video intrinsic operations to compute a
// sum of absolute differences.  The absolute differences are computed
// and the optional .add instruction is used to sum the lanes.
//
// For more information, see also the documents:
//  "Using_Inline_PTX_Assembly_In_CUDA.pdf"
// and also the PTX ISA documentation for the architecture in question, e.g.:
//  "ptx_isa_3.0K.pdf"
// included in the NVIDIA GPU Computing Toolkit
////////////////////////////////////////////////////////////////////////////////
__device__ unsigned int __usad4(unsigned int A, unsigned int B,
                                unsigned int C = 0) {
  unsigned int result;

  // Kepler (SM 3.x) and higher supports a 4 vector SAD SIMD
  asm(
      "vabsdiff4.u32.u32.u32.add"
      " %0, %1, %2, %3;"
      : "=r"(result)
      : "r"(A), "r"(B), "r"(C));

  return result;
}

////////////////////////////////////////////////////////////////////////////////
//! Simple stereo disparity kernel to test atomic instructions
//! Algorithm Explanation:
//! For stereo disparity this performs a basic block matching scheme.
//! The sum of abs. diffs between and area of the candidate pixel in the left
//! images
//! is computed against different horizontal shifts of areas from the right.
//! The shift at which the difference is minimum is taken as how far that pixel
//! moved between left/right image pairs.   The recovered motion is the
//! disparity map
//! More motion indicates more parallax indicates a closer object.
//! @param g_img1  image 1 in global memory, RGBA, 4 bytes/pixel
//! @param g_img2  image 2 in global memory
//! @param g_odata disparity map output in global memory,  unsigned int
//! output/pixel
//! @param w image width in pixels
//! @param h image height in pixels
//! @param minDisparity leftmost search range
//! @param maxDisparity rightmost search range
////////////////////////////////////////////////////////////////////////////////
__global__ void stereoDisparityKernel(unsigned int *g_img0,
                                      unsigned int *g_img1,
                                      unsigned int *g_odata, int w, int h,
                                      int minDisparity, int maxDisparity,
                                      cudaTextureObject_t tex2Dleft,
                                      cudaTextureObject_t tex2Dright) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // access thread id
  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int sidx = threadIdx.x + RAD;
  const unsigned int sidy = threadIdx.y + RAD;

  unsigned int imLeft;
  unsigned int imRight;
  unsigned int cost;
  unsigned int bestCost = 9999999;
  unsigned int bestDisparity = 0;
  __shared__ unsigned int diff[blockSize_y + 2 * RAD][blockSize_x + 2 * RAD];

  // store needed values for left image into registers (constant indexed local
  // vars)
  unsigned int imLeftA[STEPS];
  unsigned int imLeftB[STEPS];

  for (int i = 0; i < STEPS; i++) {
    int offset = -RAD + i * RAD;
    imLeftA[i] = tex2D<unsigned int>(tex2Dleft, tidx - RAD, tidy + offset);
    imLeftB[i] =
        tex2D<unsigned int>(tex2Dleft, tidx - RAD + blockSize_x, tidy + offset);
  }

  // for a fixed camera system this could be hardcoded and loop unrolled
  for (int d = minDisparity; d <= maxDisparity; d++) {
// LEFT
#pragma unroll
    for (int i = 0; i < STEPS; i++) {
      int offset = -RAD + i * RAD;
      // imLeft = tex2D( tex2Dleft, tidx-RAD, tidy+offset );
      imLeft = imLeftA[i];
      imRight = tex2D<unsigned int>(tex2Dright, tidx - RAD + d, tidy + offset);
      cost = __usad4(imLeft, imRight);
      diff[sidy + offset][sidx - RAD] = cost;
    }

// RIGHT
#pragma unroll

    for (int i = 0; i < STEPS; i++) {
      int offset = -RAD + i * RAD;

      if (threadIdx.x < 2 * RAD) {
        // imLeft = tex2D( tex2Dleft, tidx-RAD+blockSize_x, tidy+offset );
        imLeft = imLeftB[i];
        imRight = tex2D<unsigned int>(tex2Dright, tidx - RAD + blockSize_x + d,
                                      tidy + offset);
        cost = __usad4(imLeft, imRight);
        diff[sidy + offset][sidx - RAD + blockSize_x] = cost;
      }
    }

    cg::sync(cta);

// sum cost horizontally
#pragma unroll

    for (int j = 0; j < STEPS; j++) {
      int offset = -RAD + j * RAD;
      cost = 0;
#pragma unroll

      for (int i = -RAD; i <= RAD; i++) {
        cost += diff[sidy + offset][sidx + i];
      }

      cg::sync(cta);
      diff[sidy + offset][sidx] = cost;
      cg::sync(cta);
    }

    // sum cost vertically
    cost = 0;
#pragma unroll

    for (int i = -RAD; i <= RAD; i++) {
      cost += diff[sidy + i][sidx];
    }

    // see if it is better or not
    if (cost < bestCost) {
      bestCost = cost;
      bestDisparity = d + 8;
    }

    cg::sync(cta);
  }

  if (tidy < h && tidx < w) {
    g_odata[tidy * w + tidx] = bestDisparity;
  }
}

void cpu_gold_stereo(unsigned int *img0, unsigned int *img1,
                     unsigned int *odata, int w, int h, int minDisparity,
                     int maxDisparity) {
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      unsigned int bestCost = 9999999;
      unsigned int bestDisparity = 0;

      for (int d = minDisparity; d <= maxDisparity; d++) {
        unsigned int cost = 0;

        for (int i = -RAD; i <= RAD; i++) {
          for (int j = -RAD; j <= RAD; j++) {
            // border clamping
            int yy, xx, xxd;
            yy = y + i;

            if (yy < 0) yy = 0;

            if (yy >= h) yy = h - 1;

            xx = x + j;

            if (xx < 0) xx = 0;

            if (xx >= w) xx = w - 1;

            xxd = x + j + d;

            if (xxd < 0) xxd = 0;

            if (xxd >= w) xxd = w - 1;

            // sum abs diff across components
            unsigned char *A = (unsigned char *)&img0[yy * w + xx];
            unsigned char *B = (unsigned char *)&img1[yy * w + xxd];
            unsigned int absdiff = 0;

            for (int k = 0; k < 4; k++) {
              absdiff += abs((int)(A[k] - B[k]));
            }

            cost += absdiff;
          }
        }

        if (cost < bestCost) {
          bestCost = cost;
          bestDisparity = d + 8;
        }

      }  // end for disparities

      // store to best disparity
      odata[y * w + x] = bestDisparity;
    }
  }
}
#endif  // #ifndef _STEREODISPARITY_KERNEL_H_
