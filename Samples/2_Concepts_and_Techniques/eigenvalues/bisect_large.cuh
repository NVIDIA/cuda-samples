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

/* Computation of eigenvalues of a small bidiagonal matrix */

#ifndef _BISECT_LARGE_CUH_
#define _BISECT_LARGE_CUH_

extern "C" {

////////////////////////////////////////////////////////////////////////////////
//! Run the kernels to compute the eigenvalues for large matrices
//! @param  input   handles to input data
//! @param  result  handles to result data
//! @param  mat_size  matrix size
//! @param  precision  desired precision of eigenvalues
//! @param  lg  lower limit of Gerschgorin interval
//! @param  ug  upper limit of Gerschgorin interval
//! @param  iterations  number of iterations (for timing)
////////////////////////////////////////////////////////////////////////////////
void computeEigenvaluesLargeMatrix(const InputData &input,
                                   const ResultDataLarge &result,
                                   const unsigned int mat_size,
                                   const float precision, const float lg,
                                   const float ug,
                                   const unsigned int iterations);

////////////////////////////////////////////////////////////////////////////////
//! Initialize variables and memory for result
//! @param  result handles to memory
//! @param  matr_size  size of the matrix
////////////////////////////////////////////////////////////////////////////////
void initResultDataLargeMatrix(ResultDataLarge &result,
                               const unsigned int mat_size);

////////////////////////////////////////////////////////////////////////////////
//! Cleanup result memory
//! @param result  handles to memory
////////////////////////////////////////////////////////////////////////////////
void cleanupResultDataLargeMatrix(ResultDataLarge &result);

////////////////////////////////////////////////////////////////////////////////
//! Process the result, that is obtain result from device and do simple sanity
//! checking
//! @param  input   handles to input data
//! @param  result  handles to result data
//! @param  mat_size  matrix size
//! @param  filename  output filename
////////////////////////////////////////////////////////////////////////////////
bool processResultDataLargeMatrix(const InputData &input,
                                  const ResultDataLarge &result,
                                  const unsigned int mat_size,
                                  const char *filename,
                                  const unsigned int user_defined,
                                  char *exec_path);
};

#endif  // #ifndef _BISECT_LARGE_CUH_
