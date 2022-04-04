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

#include "cuda_fp16.h"
#include "helper_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

#define NUM_OF_BLOCKS 128
#define NUM_OF_THREADS 128

__forceinline__ __device__ void reduceInShared_intrinsics(half2 *const v) {
  if (threadIdx.x < 64)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 64]);
  __syncthreads();
  if (threadIdx.x < 32)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 32]);
  __syncthreads();
  if (threadIdx.x < 16)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 16]);
  __syncthreads();
  if (threadIdx.x < 8)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 8]);
  __syncthreads();
  if (threadIdx.x < 4)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 4]);
  __syncthreads();
  if (threadIdx.x < 2)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 2]);
  __syncthreads();
  if (threadIdx.x < 1)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 1]);
  __syncthreads();
}

__forceinline__ __device__ void reduceInShared_native(half2 *const v) {
  if (threadIdx.x < 64) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 64];
  __syncthreads();
  if (threadIdx.x < 32) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 32];
  __syncthreads();
  if (threadIdx.x < 16) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 16];
  __syncthreads();
  if (threadIdx.x < 8) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 8];
  __syncthreads();
  if (threadIdx.x < 4) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 4];
  __syncthreads();
  if (threadIdx.x < 2) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 2];
  __syncthreads();
  if (threadIdx.x < 1) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 1];
  __syncthreads();
}

__global__ void scalarProductKernel_intrinsics(half2 const *const a,
                                               half2 const *const b,
                                               float *const results,
                                               size_t const size) {
  const int stride = gridDim.x * blockDim.x;
  __shared__ half2 shArray[NUM_OF_THREADS];

  shArray[threadIdx.x] = __float2half2_rn(0.f);
  half2 value = __float2half2_rn(0.f);

  for (int i = threadIdx.x + blockDim.x + blockIdx.x; i < size; i += stride) {
    value = __hfma2(a[i], b[i], value);
  }

  shArray[threadIdx.x] = value;
  __syncthreads();
  reduceInShared_intrinsics(shArray);

  if (threadIdx.x == 0) {
    half2 result = shArray[0];
    float f_result = __low2float(result) + __high2float(result);
    results[blockIdx.x] = f_result;
  }
}

__global__ void scalarProductKernel_native(half2 const *const a,
                                           half2 const *const b,
                                           float *const results,
                                           size_t const size) {
  const int stride = gridDim.x * blockDim.x;
  __shared__ half2 shArray[NUM_OF_THREADS];

  half2 value(0.f, 0.f);
  shArray[threadIdx.x] = value;

  for (int i = threadIdx.x + blockDim.x + blockIdx.x; i < size; i += stride) {
    value = a[i] * b[i] + value;
  }

  shArray[threadIdx.x] = value;
  __syncthreads();
  reduceInShared_native(shArray);

  if (threadIdx.x == 0) {
    half2 result = shArray[0];
    float f_result = (float)result.y + (float)result.x;
    results[blockIdx.x] = f_result;
  }
}

void generateInput(half2 *a, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    half2 temp;
    temp.x = static_cast<float>(rand() % 4);
    temp.y = static_cast<float>(rand() % 2);
    a[i] = temp;
  }
}

int main(int argc, char *argv[]) {
  srand((unsigned int)time(NULL));
  size_t size = NUM_OF_BLOCKS * NUM_OF_THREADS * 16;

  half2 *vec[2];
  half2 *devVec[2];

  float *results;
  float *devResults;

  int devID = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp devProp;
  checkCudaErrors(cudaGetDeviceProperties(&devProp, devID));

  if (devProp.major < 5 || (devProp.major == 5 && devProp.minor < 3)) {
    printf(
        "ERROR: fp16ScalarProduct requires GPU devices with compute SM 5.3 or "
        "higher.\n");
    return EXIT_WAIVED;
  }

  for (int i = 0; i < 2; ++i) {
    checkCudaErrors(cudaMallocHost((void **)&vec[i], size * sizeof *vec[i]));
    checkCudaErrors(cudaMalloc((void **)&devVec[i], size * sizeof *devVec[i]));
  }

  checkCudaErrors(
      cudaMallocHost((void **)&results, NUM_OF_BLOCKS * sizeof *results));
  checkCudaErrors(
      cudaMalloc((void **)&devResults, NUM_OF_BLOCKS * sizeof *devResults));

  for (int i = 0; i < 2; ++i) {
    generateInput(vec[i], size);
    checkCudaErrors(cudaMemcpy(devVec[i], vec[i], size * sizeof *vec[i],
                               cudaMemcpyHostToDevice));
  }

  scalarProductKernel_native<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(
      devVec[0], devVec[1], devResults, size);

  checkCudaErrors(cudaMemcpy(results, devResults,
                             NUM_OF_BLOCKS * sizeof *results,
                             cudaMemcpyDeviceToHost));

  float result_native = 0;
  for (int i = 0; i < NUM_OF_BLOCKS; ++i) {
    result_native += results[i];
  }
  printf("Result native operators\t: %f \n", result_native);

  scalarProductKernel_intrinsics<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(
      devVec[0], devVec[1], devResults, size);

  checkCudaErrors(cudaMemcpy(results, devResults,
                             NUM_OF_BLOCKS * sizeof *results,
                             cudaMemcpyDeviceToHost));

  float result_intrinsics = 0;
  for (int i = 0; i < NUM_OF_BLOCKS; ++i) {
    result_intrinsics += results[i];
  }
  printf("Result intrinsics\t: %f \n", result_intrinsics);

  printf("&&&& fp16ScalarProduct %s\n",
         (fabs(result_intrinsics - result_native) < 0.00001) ? "PASSED"
                                                             : "FAILED");

  for (int i = 0; i < 2; ++i) {
    checkCudaErrors(cudaFree(devVec[i]));
    checkCudaErrors(cudaFreeHost(vec[i]));
  }

  checkCudaErrors(cudaFree(devResults));
  checkCudaErrors(cudaFreeHost(results));

  return EXIT_SUCCESS;
}
