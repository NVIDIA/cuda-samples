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

#include "FDTD3dReference.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdio.h>

void generateRandomData(float *data, const int dimx, const int dimy,
                        const int dimz, const float lowerBound,
                        const float upperBound) {
  srand(0);

  for (int iz = 0; iz < dimz; iz++) {
    for (int iy = 0; iy < dimy; iy++) {
      for (int ix = 0; ix < dimx; ix++) {
        *data = (float)(lowerBound +
                        ((float)rand() / (float)RAND_MAX) *
                            (upperBound - lowerBound));
        ++data;
      }
    }
  }
}

void generatePatternData(float *data, const int dimx, const int dimy,
                         const int dimz, const float lowerBound,
                         const float upperBound) {
  for (int iz = 0; iz < dimz; iz++) {
    for (int iy = 0; iy < dimy; iy++) {
      for (int ix = 0; ix < dimx; ix++) {
        *data = (float)(lowerBound +
                        ((float)iz / (float)dimz) * (upperBound - lowerBound));
        ++data;
      }
    }
  }
}

bool fdtdReference(float *output, const float *input, const float *coeff,
                   const int dimx, const int dimy, const int dimz,
                   const int radius, const int timesteps) {
  const int outerDimx = dimx + 2 * radius;
  const int outerDimy = dimy + 2 * radius;
  const int outerDimz = dimz + 2 * radius;
  const size_t volumeSize = outerDimx * outerDimy * outerDimz;
  const int stride_y = outerDimx;
  const int stride_z = stride_y * outerDimy;
  float *intermediate = 0;
  const float *bufsrc = 0;
  float *bufdst = 0;
  float *bufdstnext = 0;

  // Allocate temporary buffer
  printf(" calloc intermediate\n");
  intermediate = (float *)calloc(volumeSize, sizeof(float));

  // Decide which buffer to use first (result should end up in output)
  if ((timesteps % 2) == 0) {
    bufsrc = input;
    bufdst = intermediate;
    bufdstnext = output;
  } else {
    bufsrc = input;
    bufdst = output;
    bufdstnext = intermediate;
  }

  // Run the FDTD (naive method)
  printf(" Host FDTD loop\n");

  for (int it = 0; it < timesteps; it++) {
    printf("\tt = %d\n", it);
    const float *src = bufsrc;
    float *dst = bufdst;

    for (int iz = -radius; iz < dimz + radius; iz++) {
      for (int iy = -radius; iy < dimy + radius; iy++) {
        for (int ix = -radius; ix < dimx + radius; ix++) {
          if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 &&
              iz < dimz) {
            float value = (*src) * coeff[0];

            for (int ir = 1; ir <= radius; ir++) {
              value += coeff[ir] * (*(src + ir) + *(src - ir));  // horizontal
              value += coeff[ir] * (*(src + ir * stride_y) +
                                    *(src - ir * stride_y));  // vertical
              value +=
                  coeff[ir] * (*(src + ir * stride_z) +
                               *(src - ir * stride_z));  // in front & behind
            }

            *dst = value;
          } else {
            *dst = *src;
          }

          ++dst;
          ++src;
        }
      }
    }

    // Rotate buffers
    float *tmp = bufdst;
    bufdst = bufdstnext;
    bufdstnext = tmp;
    bufsrc = (const float *)tmp;
  }

  printf("\n");

  if (intermediate) free(intermediate);

  return true;
}

bool compareData(const float *output, const float *reference, const int dimx,
                 const int dimy, const int dimz, const int radius,
                 const float tolerance) {
  for (int iz = -radius; iz < dimz + radius; iz++) {
    for (int iy = -radius; iy < dimy + radius; iy++) {
      for (int ix = -radius; ix < dimx + radius; ix++) {
        if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 &&
            iz < dimz) {
          // Determine the absolute difference
          float difference = fabs(*reference - *output);
          float error;

          // Determine the relative error
          if (*reference != 0)
            error = difference / *reference;
          else
            error = difference;

          // Check the error is within the tolerance
          if (error > tolerance) {
            printf("Data error at point (%d,%d,%d)\t%f instead of %f\n", ix, iy,
                   iz, *output, *reference);
            return false;
          }
        }

        ++output;
        ++reference;
      }
    }
  }

  return true;
}
