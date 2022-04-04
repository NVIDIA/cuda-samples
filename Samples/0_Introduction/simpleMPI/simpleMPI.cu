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

/* Simple example demonstrating how to use MPI with CUDA
*
*  Generate some random numbers on one node.
*  Dispatch them to all nodes.
*  Compute their square root on each node's GPU.
*  Compute the average of the results using MPI.
*
*  simpleMPI.cu: GPU part, compiled with nvcc
*/

#include <iostream>
using std::cerr;
using std::endl;

#include "simpleMPI.h"

// Error handling macro
#define CUDA_CHECK(call)                                                 \
  if ((call) != cudaSuccess) {                                           \
    cudaError_t err = cudaGetLastError();                                \
    cerr << "CUDA error calling \"" #call "\", code is " << err << endl; \
    my_abort(err);                                                       \
  }

// Device code
// Very simple GPU Kernel that computes square roots of input numbers
__global__ void simpleMPIKernel(float *input, float *output) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  output[tid] = sqrt(input[tid]);
}

// Initialize an array with random data (between 0 and 1)
void initData(float *data, int dataSize) {
  for (int i = 0; i < dataSize; i++) {
    data[i] = (float)rand() / RAND_MAX;
  }
}

// CUDA computation on each node
// No MPI here, only CUDA
void computeGPU(float *hostData, int blockSize, int gridSize) {
  int dataSize = blockSize * gridSize;

  // Allocate data on GPU memory
  float *deviceInputData = NULL;
  CUDA_CHECK(cudaMalloc((void **)&deviceInputData, dataSize * sizeof(float)));

  float *deviceOutputData = NULL;
  CUDA_CHECK(cudaMalloc((void **)&deviceOutputData, dataSize * sizeof(float)));

  // Copy to GPU memory
  CUDA_CHECK(cudaMemcpy(deviceInputData, hostData, dataSize * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Run kernel
  simpleMPIKernel<<<gridSize, blockSize>>>(deviceInputData, deviceOutputData);

  // Copy data back to CPU memory
  CUDA_CHECK(cudaMemcpy(hostData, deviceOutputData, dataSize * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Free GPU memory
  CUDA_CHECK(cudaFree(deviceInputData));
  CUDA_CHECK(cudaFree(deviceOutputData));
}

float sum(float *data, int size) {
  float accum = 0.f;

  for (int i = 0; i < size; i++) {
    accum += data[i];
  }

  return accum;
}
