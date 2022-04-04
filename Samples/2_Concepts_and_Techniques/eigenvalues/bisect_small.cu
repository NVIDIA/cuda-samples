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

/* Computation of eigenvalues of a small symmetric, tridiagonal matrix */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, project
#include "helper_functions.h"
#include "helper_cuda.h"
#include "config.h"
#include "structs.h"
#include "matlab.h"

// includes, kernels
#include "bisect_kernel_small.cuh"

// includes, file
#include "bisect_small.cuh"

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
                                   const unsigned int iterations) {
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  for (unsigned int i = 0; i < iterations; ++i) {
    dim3 blocks(1, 1, 1);
    dim3 threads(MAX_THREADS_BLOCK_SMALL_MATRIX, 1, 1);

    bisectKernel<<<blocks, threads>>>(input.g_a, input.g_b, mat_size,
                                      result.g_left, result.g_right,
                                      result.g_left_count, result.g_right_count,
                                      lg, ug, 0, mat_size, precision);
  }

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  getLastCudaError("Kernel launch failed");
  printf("Average time: %f ms (%i iterations)\n",
         sdkGetTimerValue(&timer) / (float)iterations, iterations);

  sdkDeleteTimer(&timer);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize variables and memory for the result for small matrices
//! @param result  handles to the necessary memory
//! @param  mat_size  matrix_size
////////////////////////////////////////////////////////////////////////////////
void initResultSmallMatrix(ResultDataSmall &result,
                           const unsigned int mat_size) {
  result.mat_size_f = sizeof(float) * mat_size;
  result.mat_size_ui = sizeof(unsigned int) * mat_size;

  result.eigenvalues = (float *)malloc(result.mat_size_f);

  // helper variables
  result.zero_f = (float *)malloc(result.mat_size_f);
  result.zero_ui = (unsigned int *)malloc(result.mat_size_ui);

  for (unsigned int i = 0; i < mat_size; ++i) {
    result.zero_f[i] = 0.0f;
    result.zero_ui[i] = 0;

    result.eigenvalues[i] = 0.0f;
  }

  checkCudaErrors(cudaMalloc((void **)&result.g_left, result.mat_size_f));
  checkCudaErrors(cudaMalloc((void **)&result.g_right, result.mat_size_f));

  checkCudaErrors(
      cudaMalloc((void **)&result.g_left_count, result.mat_size_ui));
  checkCudaErrors(
      cudaMalloc((void **)&result.g_right_count, result.mat_size_ui));

  // initialize result memory
  checkCudaErrors(cudaMemcpy(result.g_left, result.zero_f, result.mat_size_f,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(result.g_right, result.zero_f, result.mat_size_f,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(result.g_right_count, result.zero_ui,
                             result.mat_size_ui, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(result.g_left_count, result.zero_ui,
                             result.mat_size_ui, cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////////////////////////
//! Cleanup memory and variables for result for small matrices
//! @param  result  handle to variables
////////////////////////////////////////////////////////////////////////////////
void cleanupResultSmallMatrix(ResultDataSmall &result) {
  freePtr(result.eigenvalues);
  freePtr(result.zero_f);
  freePtr(result.zero_ui);

  checkCudaErrors(cudaFree(result.g_left));
  checkCudaErrors(cudaFree(result.g_right));
  checkCudaErrors(cudaFree(result.g_left_count));
  checkCudaErrors(cudaFree(result.g_right_count));
}

////////////////////////////////////////////////////////////////////////////////
//! Process the result obtained on the device, that is transfer to host and
//! perform basic sanity checking
//! @param  input  handles to input data
//! @param  result  handles to result data
//! @param  mat_size   matrix size
//! @param  filename  output filename
////////////////////////////////////////////////////////////////////////////////
void processResultSmallMatrix(const InputData &input,
                              const ResultDataSmall &result,
                              const unsigned int mat_size,
                              const char *filename) {
  const unsigned int mat_size_f = sizeof(float) * mat_size;
  const unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;

  // copy data back to host
  float *left = (float *)malloc(mat_size_f);
  unsigned int *left_count = (unsigned int *)malloc(mat_size_ui);

  checkCudaErrors(
      cudaMemcpy(left, result.g_left, mat_size_f, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(left_count, result.g_left_count, mat_size_ui,
                             cudaMemcpyDeviceToHost));

  float *eigenvalues = (float *)malloc(mat_size_f);

  for (unsigned int i = 0; i < mat_size; ++i) {
    eigenvalues[left_count[i]] = left[i];
  }

  // save result in matlab format
  writeTridiagSymMatlab(filename, input.a, input.b + 1, eigenvalues, mat_size);

  freePtr(left);
  freePtr(left_count);
  freePtr(eigenvalues);
}
