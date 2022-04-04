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
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */

#include <helper_functions.h>  // helper functions for string parsing
#include <helper_cuda.h>  // helper functions CUDA error checking and initialization

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(float *h_CallResult, float *h_PutResult,
                                float *h_StockPrice, float *h_OptionStrike,
                                float *h_OptionYears, float Riskfree,
                                float Volatility, int optN);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 4000000;
const int NUM_ITERATIONS = 512;

const int OPT_SZ = OPT_N * sizeof(float);
const float RISKFREE = 0.02f;
const float VOLATILITY = 0.30f;

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // Start logs
  printf("[%s] - Starting...\n", argv[0]);

  //'h_' prefix - CPU (host) memory space
  float
      // Results calculated by CPU for reference
      *h_CallResultCPU,
      *h_PutResultCPU,
      // CPU copy of GPU results
      *h_CallResultGPU, *h_PutResultGPU,
      // CPU instance of input data
      *h_StockPrice, *h_OptionStrike, *h_OptionYears;

  //'d_' prefix - GPU (device) memory space
  float
      // Results calculated by GPU
      *d_CallResult,
      *d_PutResult,
      // GPU instance of input data
      *d_StockPrice, *d_OptionStrike, *d_OptionYears;

  double delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;

  StopWatchInterface *hTimer = NULL;
  int i;

  findCudaDevice(argc, (const char **)argv);

  sdkCreateTimer(&hTimer);

  printf("Initializing data...\n");
  printf("...allocating CPU memory for options.\n");
  h_CallResultCPU = (float *)malloc(OPT_SZ);
  h_PutResultCPU = (float *)malloc(OPT_SZ);
  h_CallResultGPU = (float *)malloc(OPT_SZ);
  h_PutResultGPU = (float *)malloc(OPT_SZ);
  h_StockPrice = (float *)malloc(OPT_SZ);
  h_OptionStrike = (float *)malloc(OPT_SZ);
  h_OptionYears = (float *)malloc(OPT_SZ);

  printf("...allocating GPU memory for options.\n");
  checkCudaErrors(cudaMalloc((void **)&d_CallResult, OPT_SZ));
  checkCudaErrors(cudaMalloc((void **)&d_PutResult, OPT_SZ));
  checkCudaErrors(cudaMalloc((void **)&d_StockPrice, OPT_SZ));
  checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
  checkCudaErrors(cudaMalloc((void **)&d_OptionYears, OPT_SZ));

  printf("...generating input data in CPU mem.\n");
  srand(5347);

  // Generate options set
  for (i = 0; i < OPT_N; i++) {
    h_CallResultCPU[i] = 0.0f;
    h_PutResultCPU[i] = -1.0f;
    h_StockPrice[i] = RandFloat(5.0f, 30.0f);
    h_OptionStrike[i] = RandFloat(1.0f, 100.0f);
    h_OptionYears[i] = RandFloat(0.25f, 10.0f);
  }

  printf("...copying input data to GPU mem.\n");
  // Copy options data to GPU memory for further processing
  checkCudaErrors(
      cudaMemcpy(d_StockPrice, h_StockPrice, OPT_SZ, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike, OPT_SZ,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_OptionYears, h_OptionYears, OPT_SZ, cudaMemcpyHostToDevice));
  printf("Data init done.\n\n");

  printf("Executing Black-Scholes GPU kernel (%i iterations)...\n",
         NUM_ITERATIONS);
  checkCudaErrors(cudaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  for (i = 0; i < NUM_ITERATIONS; i++) {
    BlackScholesGPU<<<DIV_UP((OPT_N / 2), 128), 128 /*480, 128*/>>>(
        (float2 *)d_CallResult, (float2 *)d_PutResult, (float2 *)d_StockPrice,
        (float2 *)d_OptionStrike, (float2 *)d_OptionYears, RISKFREE, VOLATILITY,
        OPT_N);
    getLastCudaError("BlackScholesGPU() execution failed\n");
  }

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;

  // Both call and put is calculated
  printf("Options count             : %i     \n", 2 * OPT_N);
  printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
  printf("Effective memory bandwidth: %f GB/s\n",
         ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
  printf("Gigaoptions per second    : %f     \n\n",
         ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

  printf(
      "BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u "
      "options, NumDevsUsed = %u, Workgroup = %u\n",
      (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime * 1e-3,
      (2 * OPT_N), 1, 128);

  printf("\nReading back GPU results...\n");
  // Read back GPU results to compare them to CPU results
  checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(
      cudaMemcpy(h_PutResultGPU, d_PutResult, OPT_SZ, cudaMemcpyDeviceToHost));

  printf("Checking the results...\n");
  printf("...running CPU calculations.\n\n");
  // Calculate options values on CPU
  BlackScholesCPU(h_CallResultCPU, h_PutResultCPU, h_StockPrice, h_OptionStrike,
                  h_OptionYears, RISKFREE, VOLATILITY, OPT_N);

  printf("Comparing the results...\n");
  // Calculate max absolute difference and L1 distance
  // between CPU and GPU results
  sum_delta = 0;
  sum_ref = 0;
  max_delta = 0;

  for (i = 0; i < OPT_N; i++) {
    ref = h_CallResultCPU[i];
    delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

    if (delta > max_delta) {
      max_delta = delta;
    }

    sum_delta += delta;
    sum_ref += fabs(ref);
  }

  L1norm = sum_delta / sum_ref;
  printf("L1 norm: %E\n", L1norm);
  printf("Max absolute error: %E\n\n", max_delta);

  printf("Shutting down...\n");
  printf("...releasing GPU memory.\n");
  checkCudaErrors(cudaFree(d_OptionYears));
  checkCudaErrors(cudaFree(d_OptionStrike));
  checkCudaErrors(cudaFree(d_StockPrice));
  checkCudaErrors(cudaFree(d_PutResult));
  checkCudaErrors(cudaFree(d_CallResult));

  printf("...releasing CPU memory.\n");
  free(h_OptionYears);
  free(h_OptionStrike);
  free(h_StockPrice);
  free(h_PutResultGPU);
  free(h_CallResultGPU);
  free(h_PutResultCPU);
  free(h_CallResultCPU);
  sdkDeleteTimer(&hTimer);
  printf("Shutdown done.\n");

  printf("\n[BlackScholes] - Test Summary\n");

  if (L1norm > 1e-6) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf(
      "\nNOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");
  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
