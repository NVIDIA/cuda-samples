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

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/utsname.h>
#endif

// Includes, system
#include <stdio.h>
#include <cassert>

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>  // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>  // helper functions for CUDA error check

const char *sampleName = "simpleAssert";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////
//! Tests assert function.
//! Thread whose id > N will print assertion failed error message.
////////////////////////////////////////////////////////////////////////////////
__global__ void testKernel(int N) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  assert(gtid < N);
}

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

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

void runTest(int argc, char **argv) {
  int Nblocks = 2;
  int Nthreads = 32;
  cudaError_t error;

#ifndef _WIN32
  utsname OS_System_Type;
  uname(&OS_System_Type);

  printf("OS_System_Type.release = %s\n", OS_System_Type.release);

  if (!strcasecmp(OS_System_Type.sysname, "Darwin")) {
    printf("simpleAssert is not current supported on Mac OSX\n\n");
    exit(EXIT_SUCCESS);
  } else {
    printf("OS Info: <%s>\n\n", OS_System_Type.version);
  }

#endif

  // This will pick the best possible CUDA capable device
  findCudaDevice(argc, (const char **)argv);

  // Kernel configuration, where a one-dimensional
  // grid and one-dimensional blocks are configured.
  dim3 dimGrid(Nblocks);
  dim3 dimBlock(Nthreads);

  printf("Launch kernel to generate assertion failures\n");
  testKernel<<<dimGrid, dimBlock>>>(60);

  // Synchronize (flushes assert output).
  printf("\n-- Begin assert output\n\n");
  error = cudaDeviceSynchronize();
  printf("\n-- End assert output\n\n");

  // Check for errors and failed asserts in asynchronous kernel launch.
  if (error == cudaErrorAssert) {
    printf(
        "Device assert failed as expected, "
        "CUDA error message is: %s\n\n",
        cudaGetErrorString(error));
  }

  testResult = error == cudaErrorAssert;
}
