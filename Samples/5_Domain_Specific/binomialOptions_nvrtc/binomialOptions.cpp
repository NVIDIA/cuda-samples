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

/*
 * This sample evaluates fair call price for a
 * given set of European options under binomial model.
 * See supplied whitepaper for more explanations.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include <helper_functions.h>

#include "binomialOptions_common.h"
#include "realtype.h"

////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for binomial tree results validation
////////////////////////////////////////////////////////////////////////////////

extern "C" void BlackScholesCall(real &callResult, TOptionData optionData);

////////////////////////////////////////////////////////////////////////////////
// Process single option on CPU
// Note that CPU code is for correctness testing only and not for benchmarking.
////////////////////////////////////////////////////////////////////////////////

extern "C" void binomialOptionsCPU(real &callResult, TOptionData optionData);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////

extern "C" void binomialOptionsGPU(real *callValue, TOptionData *optionData,
                                   int optN, int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////

real randData(real low, real high) {
  real t = (real)rand() / (real)RAND_MAX;
  return ((real)1.0 - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  printf("[%s] - Starting...\n", argv[0]);

  const int OPT_N = MAX_OPTIONS;

  TOptionData optionData[MAX_OPTIONS];
  real callValueBS[MAX_OPTIONS], callValueGPU[MAX_OPTIONS],
      callValueCPU[MAX_OPTIONS];

  real sumDelta, sumRef, gpuTime, errorVal;

  StopWatchInterface *hTimer = NULL;

  int i;

  sdkCreateTimer(&hTimer);

  printf("Generating input data...\n");

  // Generate options set
  srand(123);

  for (i = 0; i < OPT_N; i++) {
    optionData[i].S = randData(5.0f, 30.0f);
    optionData[i].X = randData(1.0f, 100.0f);
    optionData[i].T = randData(0.25f, 10.0f);
    optionData[i].R = 0.06f;
    optionData[i].V = 0.10f;

    BlackScholesCall(callValueBS[i], optionData[i]);
  }

  printf("Running GPU binomial tree...\n");

  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  binomialOptionsGPU(callValueGPU, optionData, OPT_N, argc, argv);

  sdkStopTimer(&hTimer);

  gpuTime = sdkGetTimerValue(&hTimer);

  printf("Options count            : %i     \n", OPT_N);
  printf("Time steps               : %i     \n", NUM_STEPS);
  printf("binomialOptionsGPU() time: %f msec\n", gpuTime);
  printf("Options per second       : %f     \n", OPT_N / (gpuTime * 0.001));

  printf("Running CPU binomial tree...\n");

  for (i = 0; i < OPT_N; i++) {
    binomialOptionsCPU(callValueCPU[i], optionData[i]);
  }

  printf("Comparing the results...\n");

  sumDelta = 0;
  sumRef = 0;
  printf("GPU binomial vs. Black-Scholes\n");

  for (i = 0; i < OPT_N; i++) {
    sumDelta += fabs(callValueBS[i] - callValueGPU[i]);
    sumRef += fabs(callValueBS[i]);
  }

  if (sumRef > 1E-5) {
    printf("L1 norm: %E\n", (double)(sumDelta / sumRef));
  } else {
    printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));
  }

  printf("CPU binomial vs. Black-Scholes\n");
  sumDelta = 0;
  sumRef = 0;

  for (i = 0; i < OPT_N; i++) {
    sumDelta += fabs(callValueBS[i] - callValueCPU[i]);
    sumRef += fabs(callValueBS[i]);
  }

  if (sumRef > 1E-5) {
    printf("L1 norm: %E\n", sumDelta / sumRef);
  } else {
    printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));
  }

  printf("CPU binomial vs. GPU binomial\n");
  sumDelta = 0;
  sumRef = 0;

  for (i = 0; i < OPT_N; i++) {
    sumDelta += fabs(callValueGPU[i] - callValueCPU[i]);
    sumRef += callValueCPU[i];
  }

  if (sumRef > 1E-5) {
    printf("L1 norm: %E\n", errorVal = sumDelta / sumRef);
  } else {
    printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));
  }

  printf("Shutting down...\n");

  sdkDeleteTimer(&hTimer);

  printf(
      "\nNOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");

  if (errorVal > 5e-4) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");

  exit(EXIT_SUCCESS);
}
