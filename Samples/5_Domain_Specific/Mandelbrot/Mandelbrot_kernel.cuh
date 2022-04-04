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

#include <stdio.h>
#include "helper_cuda.h"
#include "Mandelbrot_kernel.h"

// The dimensions of the thread block
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

#define ABS(n) ((n) < 0 ? -(n) : (n))

// Double single functions based on DSFUN90 package:
// http://crd.lbl.gov/~dhbailey/mpdist/index.html

// This function sets the DS number A equal to the double precision floating
// point number B.
inline void dsdeq(float &a0, float &a1, double b) {
  a0 = (float)b;
  a1 = (float)(b - a0);
}  // dsdcp

// This function sets the DS number A equal to the single precision floating
// point number B.
__device__ inline void dsfeq(float &a0, float &a1, float b) {
  a0 = b;
  a1 = 0.0f;
}  // dsfeq

// This function computes c = a + b.
__device__ inline void dsadd(float &c0, float &c1, const float a0,
                             const float a1, const float b0, const float b1) {
  // Compute dsa + dsb using Knuth's trick.
  float t1 = a0 + b0;
  float e = t1 - a0;
  float t2 = ((b0 - e) + (a0 - (t1 - e))) + a1 + b1;

  // The result is t1 + t2, after normalization.
  c0 = e = t1 + t2;
  c1 = t2 - (e - t1);
}  // dsadd

// This function computes c = a - b.
__device__ inline void dssub(float &c0, float &c1, const float a0,
                             const float a1, const float b0, const float b1) {
  // Compute dsa - dsb using Knuth's trick.
  float t1 = a0 - b0;
  float e = t1 - a0;
  float t2 = ((-b0 - e) + (a0 - (t1 - e))) + a1 - b1;

  // The result is t1 + t2, after normalization.
  c0 = e = t1 + t2;
  c1 = t2 - (e - t1);
}  // dssub

#if 1

// This function multiplies DS numbers A and B to yield the DS product C.
__device__ inline void dsmul(float &c0, float &c1, const float a0,
                             const float a1, const float b0, const float b1) {
  // This splits dsa(1) and dsb(1) into high-order and low-order words.
  float cona = a0 * 8193.0f;
  float conb = b0 * 8193.0f;
  float sa1 = cona - (cona - a0);
  float sb1 = conb - (conb - b0);
  float sa2 = a0 - sa1;
  float sb2 = b0 - sb1;

  // Multilply a0 * b0 using Dekker's method.
  float c11 = a0 * b0;
  float c21 = (((sa1 * sb1 - c11) + sa1 * sb2) + sa2 * sb1) + sa2 * sb2;

  // Compute a0 * b1 + a1 * b0 (only high-order word is needed).
  float c2 = a0 * b1 + a1 * b0;

  // Compute (c11, c21) + c2 using Knuth's trick, also adding low-order product.
  float t1 = c11 + c2;
  float e = t1 - c11;
  float t2 = ((c2 - e) + (c11 - (t1 - e))) + c21 + a1 * b1;

  // The result is t1 + t2, after normalization.
  c0 = e = t1 + t2;
  c1 = t2 - (e - t1);
}  // dsmul

#else

// Modified double-single mul function by Norbert Juffa, NVIDIA
// uses __fmul_rn() and __fadd_rn() intrinsics which prevent FMAD merging

/* Based on: Guillaume Da Gra�a, David Defour. Implementation of Float-Float
 * Operators on Graphics Hardware. RNC'7 pp. 23-32, 2006.
 */

// This function multiplies DS numbers A and B to yield the DS product C.
__device__ inline void dsmul(float &c0, float &c1, const float a0,
                             const float a1, const float b0, const float b1) {
  // This splits dsa(1) and dsb(1) into high-order and low-order words.
  float cona = a0 * 8193.0f;
  float conb = b0 * 8193.0f;
  float sa1 = cona - (cona - a0);
  float sb1 = conb - (conb - b0);
  float sa2 = a0 - sa1;
  float sb2 = b0 - sb1;

  // Multilply a0 * b0 using Dekker's method.
  float c11 = __fmul_rn(a0, b0);
  float c21 = (((sa1 * sb1 - c11) + sa1 * sb2) + sa2 * sb1) + sa2 * sb2;

  // Compute a0 * b1 + a1 * b0 (only high-order word is needed).
  float c2 = __fmul_rn(a0, b1) + __fmul_rn(a1, b0);

  // Compute (c11, c21) + c2 using Knuth's trick, also adding low-order product.
  float t1 = c11 + c2;
  float e = t1 - c11;
  float t2 = ((c2 - e) + (c11 - (t1 - e))) + c21 + __fmul_rn(a1, b1);

  // The result is t1 + t2, after normalization.
  c0 = e = t1 + t2;
  c1 = t2 - (e - t1);
}  // dsmul

#endif

// The core Mandelbrot CUDA GPU calculation function
#if 1
// Unrolled version
template <class T>
__device__ inline int CalcMandelbrot(const T xPos, const T yPos,
                                     const T xJParam, const T yJParam,
                                     const int crunch, const bool isJulia) {
  T x, y, xx, yy;
  int i = crunch;

  T xC, yC;

  if (isJulia) {
    xC = xJParam;
    yC = yJParam;
    y = yPos;
    x = xPos;
    yy = y * y;
    xx = x * x;

  } else {
    xC = xPos;
    yC = yPos;
    y = 0;
    x = 0;
    yy = 0;
    xx = 0;
  }

  do {
    // Iteration 1
    if (xx + yy > T(4.0)) return i - 1;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 2
    if (xx + yy > T(4.0)) return i - 2;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 3
    if (xx + yy > T(4.0)) return i - 3;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 4
    if (xx + yy > T(4.0)) return i - 4;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 5
    if (xx + yy > T(4.0)) return i - 5;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 6
    if (xx + yy > T(4.0)) return i - 6;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 7
    if (xx + yy > T(4.0)) return i - 7;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 8
    if (xx + yy > T(4.0)) return i - 8;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 9
    if (xx + yy > T(4.0)) return i - 9;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 10
    if (xx + yy > T(4.0)) return i - 10;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 11
    if (xx + yy > T(4.0)) return i - 11;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 12
    if (xx + yy > T(4.0)) return i - 12;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 13
    if (xx + yy > T(4.0)) return i - 13;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 14
    if (xx + yy > T(4.0)) return i - 14;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 15
    if (xx + yy > T(4.0)) return i - 15;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 16
    if (xx + yy > T(4.0)) return i - 16;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 17
    if (xx + yy > T(4.0)) return i - 17;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 18
    if (xx + yy > T(4.0)) return i - 18;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 19
    if (xx + yy > T(4.0)) return i - 19;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;

    // Iteration 20
    i -= 20;

    if ((i <= 0) || (xx + yy > T(4.0))) return i;

    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;
  } while (1);
}  // CalcMandelbrot

#else

template <class T>
__device__ inline int CalcMandelbrot(const T xPos, const T yPos,
                                     const T xJParam, const T yJParam,
                                     const int crunch, const isJulia) {
  T x, y, xx, yy, xC, yC;

  if (isJulia) {
    xC = xJParam;
    yC = yJParam;
    y = yPos;
    x = xPos;
    yy = y * y;
    xx = x * x;

  } else {
    xC = xPos;
    yC = yPos;
    y = 0;
    x = 0;
    yy = 0;
    xx = 0;
  }

  int i = crunch;

  while (--i && (xx + yy < T(4.0))) {
    y = x * y * T(2.0) + yC;
    x = xx - yy + xC;
    yy = y * y;
    xx = x * x;
  }

  return i;  // i > 0 ? crunch - i : 0;
}  // CalcMandelbrot

#endif

// The core Mandelbrot calculation function in double-single precision
__device__ inline int CalcMandelbrotDS(const float xPos0, const float xPos1,
                                       const float yPos0, const float yPos1,
                                       const float xJParam, const float yJParam,
                                       const int crunch, const bool isJulia) {
  float xx0, xx1;
  float yy0, yy1;
  float sum0, sum1;
  int i = crunch;

  float x0, x1, y0, y1;
  float xC0, xC1, yC0, yC1;

  if (isJulia) {
    xC0 = xJParam;
    xC1 = 0;
    yC0 = yJParam;
    yC1 = 0;
    y0 = yPos0;  // y = yPos;
    y1 = yPos1;
    x0 = xPos0;  // x = xPos;
    x1 = xPos1;
    dsmul(yy0, yy1, y0, y1, y0, y1);  // yy = y * y;
    dsmul(xx0, xx1, x0, x1, x0, x1);  // xx = x * x;
  } else {
    xC0 = xPos0;
    xC1 = xPos1;
    yC0 = yPos0;
    yC1 = yPos1;
    y0 = 0;  // y = 0 ;
    y1 = 0;
    x0 = 0;  // x = 0 ;
    x1 = 0;
    yy0 = 0;  // yy = 0 ;
    yy1 = 0;
    xx0 = 0;  // xx = 0 ;
    xx1 = 0;
  }

  dsadd(sum0, sum1, xx0, xx1, yy0, yy1);  // sum = xx + yy;

  while (--i && (sum0 + sum1 < 4.0f)) {
    dsmul(y0, y1, x0, x1, y0, y1);  // y = x * y * 2.0f + yC;  // yC is yPos for
                                    // Mandelbrot and it is yJParam for Julia
    dsadd(y0, y1, y0, y1, y0, y1);
    dsadd(y0, y1, y0, y1, yC0, yC1);

    dssub(x0, x1, xx0, xx1, yy0, yy1);  //  x = xx - yy + xC;  // xC is xPos for
                                        //  Mandelbrot and it is xJParam for
                                        //  Julia
    dsadd(x0, x1, x0, x1, xC0, xC1);

    dsmul(yy0, yy1, y0, y1, y0, y1);        // yy = y * y;
    dsmul(xx0, xx1, x0, x1, x0, x1);        // xx = x * x;
    dsadd(sum0, sum1, xx0, xx1, yy0, yy1);  // sum = xx + yy;
  }

  return i;
}  // CalcMandelbrotDS

// Determine if two pixel colors are within tolerance
__device__ inline int CheckColors(const uchar4 &color0, const uchar4 &color1) {
  int x = color1.x - color0.x;
  int y = color1.y - color0.y;
  int z = color1.z - color0.z;
  return (ABS(x) > 10) || (ABS(y) > 10) || (ABS(z) > 10);
}  // CheckColors

// Increase the grid size by 1 if the image width or height does not divide
// evenly
// by the thread block dimensions
inline int iDivUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}  // iDivUp
