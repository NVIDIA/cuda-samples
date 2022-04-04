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

// System includes.
#include <stdio.h>
#include <iostream>

// STL.
#include <vector>

// CUDA runtime.
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA.
#include <helper_functions.h>
#include <helper_cuda.h>

// Device library includes.
#include "simpleDeviceLibrary.cuh"

using std::cout;
using std::endl;

using std::vector;

#define EPS 1e-5

typedef unsigned int uint;
typedef float (*deviceFunc)(float);

const char *sampleName = "simpleSeparateCompilation";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Static device pointers to __device__ functions.
__device__ deviceFunc dMultiplyByTwoPtr = multiplyByTwo;
__device__ deviceFunc dDivideByTwoPtr = divideByTwo;

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////
//! Transforms vector.
//! Applies the __device__ function "f" to each element of the vector "v".
////////////////////////////////////////////////////////////////////////////////
__global__ void transformVector(float *v, deviceFunc f, uint size) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    v[tid] = (*f)(v[tid]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, const char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  cout << sampleName << " starting..." << endl;

  runTest(argc, (const char **)argv);

  cout << sampleName << " completed, returned " << (testResult ? "OK" : "ERROR")
       << endl;

  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void runTest(int argc, const char **argv) {
  try {
    // This will pick the best possible CUDA capable device.
    findCudaDevice(argc, (const char **)argv);

    // Create host vector.
    const uint kVectorSize = 1000;

    vector<float> hVector(kVectorSize);

    for (uint i = 0; i < kVectorSize; ++i) {
      hVector[i] = rand() / static_cast<float>(RAND_MAX);
    }

    // Create and populate device vector.
    float *dVector;
    checkCudaErrors(cudaMalloc(&dVector, kVectorSize * sizeof(float)));

    checkCudaErrors(cudaMemcpy(dVector, &hVector[0],
                               kVectorSize * sizeof(float),
                               cudaMemcpyHostToDevice));

    // Kernel configuration, where a one-dimensional
    // grid and one-dimensional blocks are configured.
    const int nThreads = 1024;
    const int nBlocks = 1;

    dim3 dimGrid(nBlocks);
    dim3 dimBlock(nThreads);

    // Test library functions.
    deviceFunc hFunctionPtr;

    cudaMemcpyFromSymbol(&hFunctionPtr, dMultiplyByTwoPtr, sizeof(deviceFunc));
    transformVector<<<dimGrid, dimBlock>>>(dVector, hFunctionPtr, kVectorSize);
    checkCudaErrors(cudaGetLastError());

    cudaMemcpyFromSymbol(&hFunctionPtr, dDivideByTwoPtr, sizeof(deviceFunc));
    transformVector<<<dimGrid, dimBlock>>>(dVector, hFunctionPtr, kVectorSize);
    checkCudaErrors(cudaGetLastError());

    // Download results.
    vector<float> hResultVector(kVectorSize);

    checkCudaErrors(cudaMemcpy(&hResultVector[0], dVector,
                               kVectorSize * sizeof(float),
                               cudaMemcpyDeviceToHost));

    // Check results.
    for (int i = 0; i < kVectorSize; ++i) {
      if (fabs(hVector[i] - hResultVector[i]) > EPS) {
        cout << "Computations were incorrect..." << endl;
        testResult = false;
        break;
      }
    }

    // Free resources.
    if (dVector) checkCudaErrors(cudaFree(dVector));
  } catch (...) {
    cout << "Error occured, exiting..." << endl;

    exit(EXIT_FAILURE);
  }
}
