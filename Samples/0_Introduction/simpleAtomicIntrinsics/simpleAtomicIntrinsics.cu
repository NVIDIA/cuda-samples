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

/* A simple program demonstrating trivial use of global memory atomic
 * device functions (atomic*() functions).
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>  // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>  // helper functions for CUDA error check

// Includes, kernels
#include "simpleAtomicIntrinsics_kernel.cuh"

const char *sampleName = "simpleAtomicIntrinsics";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

extern "C" bool computeGold(int *gpuData, const int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("%s starting...\n", sampleName);

  runTest(argc, argv);

  printf("%s completed, returned %s\n", sampleName,
         testResult ? "OK" : "ERROR!");
  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  cudaStream_t stream;
  // This will pick the best possible CUDA capable device
  findCudaDevice(argc, (const char **)argv);

  StopWatchInterface *timer;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  unsigned int numData = 11;
  unsigned int memSize = sizeof(int) * numData;

  // allocate mem for the result on host side
  int *hOData;
  checkCudaErrors(cudaMallocHost(&hOData, memSize));

  // initialize the memory
  for (unsigned int i = 0; i < numData; i++) hOData[i] = 0;

  // To make the AND and XOR tests generate something other than 0...
  hOData[8] = hOData[10] = 0xff;

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  // allocate device memory for result
  int *dOData;
  checkCudaErrors(cudaMalloc((void **)&dOData, memSize));
  // copy host memory to device to initialize to zero
  checkCudaErrors(
      cudaMemcpyAsync(dOData, hOData, memSize, cudaMemcpyHostToDevice, stream));

  // execute the kernel
  testKernel<<<numBlocks, numThreads, 0, stream>>>(dOData);

  // Copy result from device to host
  checkCudaErrors(
      cudaMemcpyAsync(hOData, dOData, memSize, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  sdkStopTimer(&timer);
  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  sdkDeleteTimer(&timer);

  // Compute reference solution
  testResult = computeGold(hOData, numThreads * numBlocks);

  // Cleanup memory
  checkCudaErrors(cudaFreeHost(hOData));
  checkCudaErrors(cudaFree(dOData));
}
