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

#ifndef _BISECT_SMALL_CUH_
#define _BISECT_SMALL_CUH_

extern "C" {

////////////////////////////////////////////////////////////////////////////////
//! Determine eigenvalues for matrices smaller than MAX_SMALL_MATRIX
//! @param TimingIterations  number of iterations for timing
//! @param  input  handles to input data of kernel
//! @param  result handles to result of kernel
//! @param  mat_size  matrix size
//! @param  lg  lower limit of Gerschgorin interval
//! @param  ug  upper limit of Gerschgorin interval
//! @param  precision  desired precision of eigenvalues
//! @param  iterations  number of iterations for timing
////////////////////////////////////////////////////////////////////////////////
void computeEigenvaluesSmallMatrix(const InputData &input,
                                   ResultDataSmall &result,
                                   const unsigned int mat_size, const float lg,
                                   const float ug, const float precision,
                                   const unsigned int iterations);

////////////////////////////////////////////////////////////////////////////////
//! Initialize variables and memory for the result for small matrices
//! @param result  handles to the necessary memory
//! @param  mat_size  matrix_size
////////////////////////////////////////////////////////////////////////////////
void initResultSmallMatrix(ResultDataSmall &result,
                           const unsigned int mat_size);

////////////////////////////////////////////////////////////////////////////////
//! Cleanup memory and variables for result for small matrices
//! @param  result  handle to variables
////////////////////////////////////////////////////////////////////////////////
void cleanupResultSmallMatrix(ResultDataSmall &result);

////////////////////////////////////////////////////////////////////////////////
//! Process the result obtained on the device, that is transfer to host and
//! perform basic sanity checking
//! @param  input   handles to input data
//! @param  result  handles to result variables
//! @param  mat_size   matrix size
//! @param  filename  output filename
////////////////////////////////////////////////////////////////////////////////
void processResultSmallMatrix(const InputData &input,
                              const ResultDataSmall &result,
                              const unsigned int mat_size,
                              const char *filename);
}

#endif  // #ifndef _BISECT_SMALL_CUH_
