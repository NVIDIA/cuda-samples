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

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "scan_common.h"

int main(int argc, char **argv) {
  printf("%s Starting...\n\n", argv[0]);

  // Use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  findCudaDevice(argc, (const char **)argv);

  uint *d_Input, *d_Output;
  uint *h_Input, *h_OutputCPU, *h_OutputGPU;
  StopWatchInterface *hTimer = NULL;
  const uint N = 13 * 1048576 / 2;

  printf("Allocating and initializing host arrays...\n");
  sdkCreateTimer(&hTimer);
  h_Input = (uint *)malloc(N * sizeof(uint));
  h_OutputCPU = (uint *)malloc(N * sizeof(uint));
  h_OutputGPU = (uint *)malloc(N * sizeof(uint));
  srand(2009);

  for (uint i = 0; i < N; i++) {
    h_Input[i] = rand();
  }

  printf("Allocating and initializing CUDA arrays...\n");
  checkCudaErrors(cudaMalloc((void **)&d_Input, N * sizeof(uint)));
  checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));
  checkCudaErrors(
      cudaMemcpy(d_Input, h_Input, N * sizeof(uint), cudaMemcpyHostToDevice));

  printf("Initializing CUDA-C scan...\n\n");
  initScan();

  int globalFlag = 1;
  size_t szWorkgroup;
  const int iCycles = 100;
  printf(
      "*** Running GPU scan for short arrays (%d identical iterations)...\n\n",
      iCycles);

  for (uint arrayLength = MIN_SHORT_ARRAY_SIZE;
       arrayLength <= MAX_SHORT_ARRAY_SIZE; arrayLength <<= 1) {
    printf("Running scan for %u elements (%u arrays)...\n", arrayLength,
           N / arrayLength);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (int i = 0; i < iCycles; i++) {
      szWorkgroup =
          scanExclusiveShort(d_Output, d_Input, N / arrayLength, arrayLength);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double timerValue = 1.0e-3 * sdkGetTimerValue(&hTimer) / iCycles;

    printf("Validating the results...\n");
    printf("...reading back GPU results\n");
    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint),
                               cudaMemcpyDeviceToHost));

    printf(" ...scanExclusiveHost()\n");
    scanExclusiveHost(h_OutputCPU, h_Input, N / arrayLength, arrayLength);

    // Compare GPU results with CPU results and accumulate error for this test
    printf(" ...comparing the results\n");
    int localFlag = 1;

    for (uint i = 0; i < N; i++) {
      if (h_OutputCPU[i] != h_OutputGPU[i]) {
        localFlag = 0;
        break;
      }
    }

    // Log message on individual test result, then accumulate to global flag
    printf(" ...Results %s\n\n",
           (localFlag == 1) ? "Match" : "DON'T Match !!!");
    globalFlag = globalFlag && localFlag;

    // Data log
    if (arrayLength == MAX_SHORT_ARRAY_SIZE) {
      printf("\n");
      printf(
          "scan, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u "
          "Elements, NumDevsUsed = %u, Workgroup = %u\n",
          (1.0e-6 * (double)arrayLength / timerValue), timerValue,
          (unsigned int)arrayLength, 1, (unsigned int)szWorkgroup);
      printf("\n");
    }
  }

  printf(
      "***Running GPU scan for large arrays (%u identical iterations)...\n\n",
      iCycles);

  for (uint arrayLength = MIN_LARGE_ARRAY_SIZE;
       arrayLength <= MAX_LARGE_ARRAY_SIZE; arrayLength <<= 1) {
    printf("Running scan for %u elements (%u arrays)...\n", arrayLength,
           N / arrayLength);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (int i = 0; i < iCycles; i++) {
      szWorkgroup =
          scanExclusiveLarge(d_Output, d_Input, N / arrayLength, arrayLength);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double timerValue = 1.0e-3 * sdkGetTimerValue(&hTimer) / iCycles;

    printf("Validating the results...\n");
    printf("...reading back GPU results\n");
    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint),
                               cudaMemcpyDeviceToHost));

    printf("...scanExclusiveHost()\n");
    scanExclusiveHost(h_OutputCPU, h_Input, N / arrayLength, arrayLength);

    // Compare GPU results with CPU results and accumulate error for this test
    printf(" ...comparing the results\n");
    int localFlag = 1;

    for (uint i = 0; i < N; i++) {
      if (h_OutputCPU[i] != h_OutputGPU[i]) {
        localFlag = 0;
        break;
      }
    }

    // Log message on individual test result, then accumulate to global flag
    printf(" ...Results %s\n\n",
           (localFlag == 1) ? "Match" : "DON'T Match !!!");
    globalFlag = globalFlag && localFlag;

    // Data log
    if (arrayLength == MAX_LARGE_ARRAY_SIZE) {
      printf("\n");
      printf(
          "scan, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u "
          "Elements, NumDevsUsed = %u, Workgroup = %u\n",
          (1.0e-6 * (double)arrayLength / timerValue), timerValue,
          (unsigned int)arrayLength, 1, (unsigned int)szWorkgroup);
      printf("\n");
    }
  }

  printf("Shutting down...\n");
  closeScan();
  checkCudaErrors(cudaFree(d_Output));
  checkCudaErrors(cudaFree(d_Input));

  sdkDeleteTimer(&hTimer);

  // pass or fail (cumulative... all tests in the loop)
  exit(globalFlag ? EXIT_SUCCESS : EXIT_FAILURE);
}
