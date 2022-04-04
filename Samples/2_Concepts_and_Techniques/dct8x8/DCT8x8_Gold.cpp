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
* \file DCT8x8_Gold.cpp
* \brief Contains DCT, IDCT and quantization routines, used in JPEG internal
* data processing. Host code.
*
* This sample implements forward and inverse Discrete Cosine Transform to blocks
* of image pixels (of 8x8 size), as in JPEG standard. The data processing is
*done
* using floating point representation.
* The routine that performs quantization of coefficients is also included.
*/

#include "Common.h"
#include "BmpUtil.h"

/**
*  This unitary matrix performs DCT of rows of the matrix to the left
*/
const float DCTv8matrix[BLOCK_SIZE2] = {
  0.3535533905932738f,  0.4903926402016152f,  0.4619397662556434f,  0.4157348061512726f,  0.3535533905932738f,  0.2777851165098011f,  0.1913417161825449f,  0.0975451610080642f,
  0.3535533905932738f,  0.4157348061512726f,  0.1913417161825449f, -0.0975451610080641f, -0.3535533905932737f, -0.4903926402016152f, -0.4619397662556434f, -0.2777851165098011f,
  0.3535533905932738f,  0.2777851165098011f, -0.1913417161825449f, -0.4903926402016152f, -0.3535533905932738f,  0.0975451610080642f,  0.4619397662556433f,  0.4157348061512727f,
  0.3535533905932738f,  0.0975451610080642f, -0.4619397662556434f, -0.2777851165098011f,  0.3535533905932737f,  0.4157348061512727f, -0.1913417161825450f, -0.4903926402016153f,
  0.3535533905932738f, -0.0975451610080641f, -0.4619397662556434f,  0.2777851165098009f,  0.3535533905932738f, -0.4157348061512726f, -0.1913417161825453f,  0.4903926402016152f,
  0.3535533905932738f, -0.2777851165098010f, -0.1913417161825452f,  0.4903926402016153f, -0.3535533905932733f, -0.0975451610080649f,  0.4619397662556437f, -0.4157348061512720f,
  0.3535533905932738f, -0.4157348061512727f,  0.1913417161825450f,  0.0975451610080640f, -0.3535533905932736f,  0.4903926402016152f, -0.4619397662556435f,  0.2777851165098022f,
  0.3535533905932738f, -0.4903926402016152f,  0.4619397662556433f, -0.4157348061512721f,  0.3535533905932733f, -0.2777851165098008f,  0.1913417161825431f, -0.0975451610080625f
};

/**
*  This unitary matrix performs DCT of columns of the matrix to the right
*/
const float DCTv8matrixT[BLOCK_SIZE2] = {
  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,
  0.4903926402016152f,  0.4157348061512726f,  0.2777851165098011f,  0.0975451610080642f, -0.0975451610080641f, -0.2777851165098010f, -0.4157348061512727f, -0.4903926402016152f,
  0.4619397662556434f,  0.1913417161825449f, -0.1913417161825449f, -0.4619397662556434f, -0.4619397662556434f, -0.1913417161825452f,  0.1913417161825450f,  0.4619397662556433f,
  0.4157348061512726f, -0.0975451610080641f, -0.4903926402016152f, -0.2777851165098011f,  0.2777851165098009f,  0.4903926402016153f,  0.0975451610080640f, -0.4157348061512721f,
  0.3535533905932738f, -0.3535533905932737f, -0.3535533905932738f,  0.3535533905932737f,  0.3535533905932738f, -0.3535533905932733f, -0.3535533905932736f,  0.3535533905932733f,
  0.2777851165098011f, -0.4903926402016152f,  0.0975451610080642f,  0.4157348061512727f, -0.4157348061512726f, -0.0975451610080649f,  0.4903926402016152f, -0.2777851165098008f,
  0.1913417161825449f, -0.4619397662556434f,  0.4619397662556433f, -0.1913417161825450f, -0.1913417161825453f,  0.4619397662556437f, -0.4619397662556435f,  0.1913417161825431f,
  0.0975451610080642f, -0.2777851165098011f,  0.4157348061512727f, -0.4903926402016153f,  0.4903926402016152f, -0.4157348061512720f,  0.2777851165098022f, -0.0975451610080625f
};

/**
*  JPEG quality=0_of_12 quantization matrix
*/
float Q[BLOCK_SIZE2] = {
  32.f,  33.f,  51.f,  81.f,  66.f,  39.f,  34.f,  17.f,
  33.f,  36.f,  48.f,  47.f,  28.f,  23.f,  12.f,  12.f,
  51.f,  48.f,  47.f,  28.f,  23.f,  12.f,  12.f,  12.f,
  81.f,  47.f,  28.f,  23.f,  12.f,  12.f,  12.f,  12.f,
  66.f,  28.f,  23.f,  12.f,  12.f,  12.f,  12.f,  12.f,
  39.f,  23.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,
  34.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,
  17.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f,  12.f
};

/**
**************************************************************************
*  Performs multiplication of two 8x8 matrices
*
* \param M1             [IN] - Pointer to the first matrix
* \param M1Stride       [IN] - Stride of the first matrix
* \param M2             [IN] - Pointer to the second matrix
* \param M2Stride       [IN] - Stride of the second matrix
* \param Mres           [OUT] - Pointer to the result matrix
* \param MresStride     [IN] - Stride of the result matrix
*
* \return None
*/
void mult8x8(const float *M1, int M1Stride, const float *M2, int M2Stride,
             float *Mres, int MresStride) {
  for (int i = 0; i < BLOCK_SIZE; i++) {
    for (int j = 0; j < BLOCK_SIZE; j++) {
      float accumul = 0;

      for (int k = 0; k < BLOCK_SIZE; k++) {
        accumul += M1[i * M1Stride + k] * M2[k * M2Stride + j];
      }

      Mres[i * MresStride + j] = accumul;
    }
  }
}

/**
**************************************************************************
*  Performs 8x8 block-wise Forward Discrete Cosine Transform of the given
*  image plane and outputs result to the plane of coefficients.
*  1st version.
*
* \param fSrc       [IN] - Source image plane
* \param fDst       [OUT] - Destination coefficients plane
* \param Stride     [IN] - Stride of both planes
* \param Size       [IN] - Size of planes
*
* \return None
*/
extern "C" void computeDCT8x8Gold1(const float *fSrc, float *fDst, int Stride,
                                   ROI Size) {
  float tmpblock[BLOCK_SIZE2];

  // perform block wise DCT
  // DCT(A) = DCTv8matrixT * A * DCTv8matrix
  for (int i = 0; i + BLOCK_SIZE - 1 < Size.height; i += BLOCK_SIZE) {
    for (int j = 0; j + BLOCK_SIZE - 1 < Size.width; j += BLOCK_SIZE) {
      // tmpblock = DCTv8matrixT * A
      mult8x8(DCTv8matrixT, BLOCK_SIZE, fSrc + i * Stride + j, Stride, tmpblock,
              BLOCK_SIZE);
      // DCT(A) = tmpblock * DCTv8matrix
      mult8x8(tmpblock, BLOCK_SIZE, DCTv8matrix, BLOCK_SIZE,
              fDst + i * Stride + j, Stride);
    }
  }
}

/**
**************************************************************************
*  Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given
*  coefficients plane and outputs result to the image plane.
*  1st version.
*
* \param fSrc       [IN] - Source coefficients plane
* \param fDst       [OUT] - Destination image plane
* \param Stride     [IN] - Stride of both planes
* \param Size       [IN] - Size of planes
*
* \return None
*/
extern "C" void computeIDCT8x8Gold1(const float *fSrc, float *fDst, int Stride,
                                    ROI Size) {
  float tmpblock[BLOCK_SIZE2];

  // perform block wise IDCT
  // IDCT(A) = DCTv8matrix * A * DCTv8matrixT
  for (int i = 0; i + BLOCK_SIZE - 1 < Size.height; i += BLOCK_SIZE) {
    for (int j = 0; j + BLOCK_SIZE - 1 < Size.width; j += BLOCK_SIZE) {
      // tmpblock = DCTv8matrix * A
      mult8x8(DCTv8matrix, BLOCK_SIZE, fSrc + i * Stride + j, Stride, tmpblock,
              BLOCK_SIZE);
      // DCT(A) = tmpblock * DCTv8matrixT;
      mult8x8(tmpblock, BLOCK_SIZE, DCTv8matrixT, BLOCK_SIZE,
              fDst + i * Stride + j, Stride);
    }
  }
}

/**
**************************************************************************
*  Performs in-place quantization of given coefficients plane using
*  predefined quantization matrices (float elements)
*
* \param fSrcDst        [IN/OUT] - Coefficients plane
* \param Stride         [IN] - Stride of SrcDst
* \param Size           [IN] - Size of the plane
*
* \return None
*/
extern "C" void quantizeGoldFloat(float *fSrcDst, int Stride, ROI Size) {
  // perform block wise in-place quantization using Q
  // Q(A) = round(A ./ Q) .* Q;
  for (int i = 0; i < Size.height; i++) {
    for (int j = 0; j < Size.width; j++) {
      int qx = j % BLOCK_SIZE;
      int qy = i % BLOCK_SIZE;
      float quantized =
          round_f(fSrcDst[i * Stride + j] / Q[(qy << BLOCK_SIZE_LOG2) + qx]);
      fSrcDst[i * Stride + j] = quantized * Q[(qy << BLOCK_SIZE_LOG2) + qx];
    }
  }
}

/**
**************************************************************************
*  Performs in-place quantization of given coefficients plane using
*  predefined quantization matrices (short elements)
*
* \param fSrcDst        [IN/OUT] - Coefficients plane
* \param Stride         [IN] - Stride of SrcDst
* \param Size           [IN] - Size of the plane
*
* \return None
*/
void quantizeGoldShort(short *fSrcDst, int Stride, ROI Size) {
  // perform block wise in-place quantization using Q
  // Q(A) = round(A ./ Q) .* Q;
  for (int i = 0; i < Size.height; i++) {
    for (int j = 0; j < Size.width; j++) {
      int qx = j % BLOCK_SIZE;
      int qy = i % BLOCK_SIZE;
      short temp = fSrcDst[i * Stride + j];
      short quant = (short)(Q[(qy << BLOCK_SIZE_LOG2) + qx]);

      if (temp < 0) {
        temp = -temp;
        temp += quant >> 1;
        temp /= quant;
        temp = -temp;
      } else {
        temp += quant >> 1;
        temp /= quant;
      }

      fSrcDst[i * Stride + j] = temp * quant;
    }
  }
}

// Used in forward and inverse DCT.
float C_a = 1.387039845322148f;  //!< a = (2^0.5) * cos(    pi / 16);
float C_b = 1.306562964876377f;  //!< b = (2^0.5) * cos(    pi /  8);
float C_c = 1.175875602419359f;  //!< c = (2^0.5) * cos(3 * pi / 16);
float C_d = 0.785694958387102f;  //!< d = (2^0.5) * cos(5 * pi / 16);
float C_e = 0.541196100146197f;  //!< e = (2^0.5) * cos(3 * pi /  8);
float C_f = 0.275899379282943f;  //!< f = (2^0.5) * cos(7 * pi / 16);

/**
*  Normalization constant that is used in forward and inverse DCT
*/
float C_norm = 0.3535533905932737f;  // 1 / (8^0.5)

/**
**************************************************************************
*  Performs DCT of vector of 8 elements.
*
* \param FirstIn        [IN] - Pointer to the first element of input vector
* \param StepIn         [IN] - Value to add to ptr to access other input
*elements
* \param FirstOut       [OUT] - Pointer to the first element of output vector
* \param StepOut        [IN] - Value to add to ptr to access other output
*elements
*
* \return None
*/
void SubroutineDCTvector(float *FirstIn, int StepIn, float *FirstOut,
                         int StepOut) {
  float X07P = FirstIn[0 * StepIn] + FirstIn[7 * StepIn];
  float X16P = FirstIn[1 * StepIn] + FirstIn[6 * StepIn];
  float X25P = FirstIn[2 * StepIn] + FirstIn[5 * StepIn];
  float X34P = FirstIn[3 * StepIn] + FirstIn[4 * StepIn];

  float X07M = FirstIn[0 * StepIn] - FirstIn[7 * StepIn];
  float X61M = FirstIn[6 * StepIn] - FirstIn[1 * StepIn];
  float X25M = FirstIn[2 * StepIn] - FirstIn[5 * StepIn];
  float X43M = FirstIn[4 * StepIn] - FirstIn[3 * StepIn];

  float X07P34PP = X07P + X34P;
  float X07P34PM = X07P - X34P;
  float X16P25PP = X16P + X25P;
  float X16P25PM = X16P - X25P;

  FirstOut[0 * StepOut] = C_norm * (X07P34PP + X16P25PP);
  FirstOut[2 * StepOut] = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
  FirstOut[4 * StepOut] = C_norm * (X07P34PP - X16P25PP);
  FirstOut[6 * StepOut] = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

  FirstOut[1 * StepOut] =
      C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
  FirstOut[3 * StepOut] =
      C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
  FirstOut[5 * StepOut] =
      C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
  FirstOut[7 * StepOut] =
      C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

/**
**************************************************************************
*  Performs IDCT of vector of 8 elements.
*
* \param FirstIn        [IN] - Pointer to the first element of input vector
* \param StepIn         [IN] - Value to add to ptr to access other input
*elements
* \param FirstOut       [OUT] - Pointer to the first element of output vector
* \param StepOut        [IN] - Value to add to ptr to access other output
*elements
*
* \return None
*/
void SubroutineIDCTvector(float *FirstIn, int StepIn, float *FirstOut,
                          int StepOut) {
  float Y04P = FirstIn[0 * StepIn] + FirstIn[4 * StepIn];
  float Y2b6eP = C_b * FirstIn[2 * StepIn] + C_e * FirstIn[6 * StepIn];

  float Y04P2b6ePP = Y04P + Y2b6eP;
  float Y04P2b6ePM = Y04P - Y2b6eP;
  float Y7f1aP3c5dPP = C_f * FirstIn[7 * StepIn] + C_a * FirstIn[1 * StepIn] +
                       C_c * FirstIn[3 * StepIn] + C_d * FirstIn[5 * StepIn];
  float Y7a1fM3d5cMP = C_a * FirstIn[7 * StepIn] - C_f * FirstIn[1 * StepIn] +
                       C_d * FirstIn[3 * StepIn] - C_c * FirstIn[5 * StepIn];

  float Y04M = FirstIn[0 * StepIn] - FirstIn[4 * StepIn];
  float Y2e6bM = C_e * FirstIn[2 * StepIn] - C_b * FirstIn[6 * StepIn];

  float Y04M2e6bMP = Y04M + Y2e6bM;
  float Y04M2e6bMM = Y04M - Y2e6bM;
  float Y1c7dM3f5aPM = C_c * FirstIn[1 * StepIn] - C_d * FirstIn[7 * StepIn] -
                       C_f * FirstIn[3 * StepIn] - C_a * FirstIn[5 * StepIn];
  float Y1d7cP3a5fMM = C_d * FirstIn[1 * StepIn] + C_c * FirstIn[7 * StepIn] -
                       C_a * FirstIn[3 * StepIn] + C_f * FirstIn[5 * StepIn];

  FirstOut[0 * StepOut] = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
  FirstOut[7 * StepOut] = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
  FirstOut[4 * StepOut] = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
  FirstOut[3 * StepOut] = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

  FirstOut[1 * StepOut] = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
  FirstOut[5 * StepOut] = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
  FirstOut[2 * StepOut] = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
  FirstOut[6 * StepOut] = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

/**
**************************************************************************
*  Performs 8x8 block-wise Forward Discrete Cosine Transform of the given
*  image plane and outputs result to the plane of coefficients.
*  2nd version.
*
* \param fSrc       [IN] - Source image plane
* \param fDst       [OUT] - Destination coefficients plane
* \param Stride     [IN] - Stride of both planes
* \param Size       [IN] - Size of planes
*
* \return None
*/
extern "C" void computeDCT8x8Gold2(const float *fSrc, float *fDst, int Stride,
                                   ROI Size) {
  for (int i = 0; i + BLOCK_SIZE - 1 < Size.height; i += BLOCK_SIZE) {
    for (int j = 0; j + BLOCK_SIZE - 1 < Size.width; j += BLOCK_SIZE) {
      // process rows
      for (int k = 0; k < BLOCK_SIZE; k++) {
        SubroutineDCTvector((float *)fSrc + (i + k) * Stride + j, 1,
                            fDst + (i + k) * Stride + j, 1);
      }

      // process columns
      for (int k = 0; k < BLOCK_SIZE; k++) {
        SubroutineDCTvector(fDst + i * Stride + (j + k), Stride,
                            fDst + i * Stride + (j + k), Stride);
      }
    }
  }
}

/**
**************************************************************************
*  Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given
*  coefficients plane and outputs result to the image plane
*  2nd version.
*
* \param fSrc       [IN] - Source coefficients plane
* \param fDst       [OUT] - Destination image plane
* \param Stride     [IN] - Stride of both planes
* \param Size       [IN] - Size of planes
*
* \return None
*/
extern "C" void computeIDCT8x8Gold2(const float *fSrc, float *fDst, int Stride,
                                    ROI Size) {
  for (int i = 0; i + BLOCK_SIZE - 1 < Size.height; i += BLOCK_SIZE) {
    for (int j = 0; j + BLOCK_SIZE - 1 < Size.width; j += BLOCK_SIZE) {
      // process rows
      for (int k = 0; k < BLOCK_SIZE; k++) {
        SubroutineIDCTvector((float *)fSrc + (i + k) * Stride + j, 1,
                             fDst + (i + k) * Stride + j, 1);
      }

      // process columns
      for (int k = 0; k < BLOCK_SIZE; k++) {
        SubroutineIDCTvector(fDst + i * Stride + (j + k), Stride,
                             fDst + i * Stride + (j + k), Stride);
      }
    }
  }
}
