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
 * This sample demonstrates the use of CURAND to generate
 * random numbers on GPU and CPU.
 */

// Utilities and system includes
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <curand.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda_runtime.h>
#include <curand.h>

float compareResults(int rand_n, float *h_RandGPU, float *h_RandCPU);

const int DEFAULT_RAND_N = 2400000;
const unsigned int DEFAULT_SEED = 777;

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // Start logs
  printf("%s Starting...\n\n", argv[0]);

  // initialize the GPU, either identified by --device
  // or by picking the device with highest flop rate.
  int devID = findCudaDevice(argc, (const char **)argv);

  // parsing the number of random numbers to generate
  int rand_n = DEFAULT_RAND_N;

  if (checkCmdLineFlag(argc, (const char **)argv, "count")) {
    rand_n = getCmdLineArgumentInt(argc, (const char **)argv, "count");
  }

  printf("Allocating data for %i samples...\n", rand_n);

  // parsing the seed
  int seed = DEFAULT_SEED;

  if (checkCmdLineFlag(argc, (const char **)argv, "seed")) {
    seed = getCmdLineArgumentInt(argc, (const char **)argv, "seed");
  }

  printf("Seeding with %i ...\n", seed);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  float *d_Rand;
  checkCudaErrors(cudaMalloc((void **)&d_Rand, rand_n * sizeof(float)));

  curandGenerator_t prngGPU;
  checkCudaErrors(curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32));
  checkCudaErrors(curandSetStream(prngGPU, stream));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prngGPU, seed));

  curandGenerator_t prngCPU;
  checkCudaErrors(
      curandCreateGeneratorHost(&prngCPU, CURAND_RNG_PSEUDO_MTGP32));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prngCPU, seed));

  //
  // Example 1: Compare random numbers generated on GPU and CPU
  float *h_RandGPU;
  checkCudaErrors(cudaMallocHost(&h_RandGPU, rand_n * sizeof(float)));

  printf("Generating random numbers on GPU...\n\n");
  checkCudaErrors(curandGenerateUniform(prngGPU, (float *)d_Rand, rand_n));

  printf("\nReading back the results...\n");
  checkCudaErrors(cudaMemcpyAsync(h_RandGPU, d_Rand, rand_n * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));

  float *h_RandCPU = (float *)malloc(rand_n * sizeof(float));

  printf("Generating random numbers on CPU...\n\n");
  checkCudaErrors(curandGenerateUniform(prngCPU, (float *)h_RandCPU, rand_n));

  checkCudaErrors(cudaStreamSynchronize(stream));
  printf("Comparing CPU/GPU random numbers...\n\n");
  float L1norm = compareResults(rand_n, h_RandGPU, h_RandCPU);

  //
  // Example 2: Timing of random number generation on GPU
  const int numIterations = 10;
  int i;
  StopWatchInterface *hTimer;

  sdkCreateTimer(&hTimer);
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  for (i = 0; i < numIterations; i++) {
    checkCudaErrors(curandGenerateUniform(prngGPU, (float *)d_Rand, rand_n));
  }

  checkCudaErrors(cudaStreamSynchronize(stream));
  sdkStopTimer(&hTimer);

  double gpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer) / (double)numIterations;

  printf(
      "MersenneTwisterGP11213, Throughput = %.4f GNumbers/s, Time = %.5f s, "
      "Size = %u Numbers\n",
      1.0e-9 * rand_n / gpuTime, gpuTime, rand_n);

  printf("Shutting down...\n");

  checkCudaErrors(curandDestroyGenerator(prngGPU));
  checkCudaErrors(curandDestroyGenerator(prngCPU));
  checkCudaErrors(cudaStreamDestroy(stream));
  checkCudaErrors(cudaFree(d_Rand));
  sdkDeleteTimer(&hTimer);
  checkCudaErrors(cudaFreeHost(h_RandGPU));
  free(h_RandCPU);

  exit(L1norm < 1e-6 ? EXIT_SUCCESS : EXIT_FAILURE);
}

float compareResults(int rand_n, float *h_RandGPU, float *h_RandCPU) {
  int i;
  float rCPU, rGPU, delta;
  float max_delta = 0.;
  float sum_delta = 0.;
  float sum_ref = 0.;

  for (i = 0; i < rand_n; i++) {
    rCPU = h_RandCPU[i];
    rGPU = h_RandGPU[i];
    delta = fabs(rCPU - rGPU);
    sum_delta += delta;
    sum_ref += fabs(rCPU);

    if (delta >= max_delta) {
      max_delta = delta;
    }
  }

  float L1norm = (float)(sum_delta / sum_ref);
  printf("Max absolute error: %E\n", max_delta);
  printf("L1 norm: %E\n\n", L1norm);

  return L1norm;
}
