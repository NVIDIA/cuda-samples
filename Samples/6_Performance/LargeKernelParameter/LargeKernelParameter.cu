/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
 * This is a simple test showing performance and usability
 * improvements with large kernel parameters introduced in CUDA 12.1
 */
#include <chrono>
#include <iostream>
#include <cassert>

// Utility includes
#include <helper_cuda.h>

using namespace std;
using namespace std::chrono;

#define TEST_ITERATIONS     (1000)
#define TOTAL_PARAMS        (8000)  // ints
#define KERNEL_PARAM_LIMIT  (1024)  // ints
#define CONST_COPIED_PARAMS (TOTAL_PARAMS - KERNEL_PARAM_LIMIT)

__constant__ int excess_params[CONST_COPIED_PARAMS];

typedef struct {
  int param[KERNEL_PARAM_LIMIT];
} param_t;

typedef struct {
  int param[TOTAL_PARAMS];
} param_large_t;

// Kernel with 4KB kernel parameter limit
__global__ void kernelDefault(__grid_constant__ const param_t p, int *result) {
  int tmp = 0;

  // accumulate kernel parameters
  for (int i = 0; i < KERNEL_PARAM_LIMIT; ++i) {
    tmp += p.param[i];
  }

  // accumulate excess values passed via const memory
  for (int i = 0; i < CONST_COPIED_PARAMS; ++i) {
    tmp += excess_params[i];
  }

  *result = tmp;
}

// Kernel with 32,764 byte kernel parameter limit
__global__ void kernelLargeParam(__grid_constant__ const param_large_t p, int *result) {
  int tmp = 0;

  // accumulate kernel parameters
  for (int i = 0; i < TOTAL_PARAMS; ++i) {
    tmp += p.param[i];
  }

  *result = tmp;
}

static void report_time(std::chrono::time_point<std::chrono::steady_clock> start,
                        std::chrono::time_point<std::chrono::steady_clock> end,
                        int iters) {
  auto usecs = duration_cast<duration<float,
                                      microseconds::period>>(end - start);
  cout << usecs.count() / iters << endl;
}

int main() {
  int rc;
  cudaFree(0);

  param_t p;
  param_large_t p_large;

  // pageable host memory that holds excess constants passed via constant memory
  int *copied_params = (int *)malloc(CONST_COPIED_PARAMS * sizeof(int));
  assert(copied_params);

  // storage for computed result
  int *d_result;
  int h_result;
  checkCudaErrors(cudaMalloc(&d_result, sizeof(int)));

  int expected_result = 0;

  // fill in data for validation
  for (int i = 0; i < KERNEL_PARAM_LIMIT; ++i) {
    p.param[i] = (i & 0xFF);
  }
  for (int i = KERNEL_PARAM_LIMIT; i < TOTAL_PARAMS; ++i) {
    copied_params[i - KERNEL_PARAM_LIMIT] = (i & 0xFF);
  }
  for (int i = 0; i < TOTAL_PARAMS; ++i) {
    p_large.param[i] = (i & 0xFF);
    expected_result += (i & 0xFF);
  }

  // warmup, verify correctness
  checkCudaErrors(cudaMemcpyToSymbol(excess_params, copied_params, CONST_COPIED_PARAMS * sizeof(int), 0, cudaMemcpyHostToDevice));
  kernelDefault<<<1,1>>>(p, d_result);
  checkCudaErrors(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaDeviceSynchronize());
  if(h_result != expected_result) {
    std::cout << "Test failed" << std::endl;
	 rc=-1;
	 goto Exit;    
  }

  kernelLargeParam<<<1,1>>>(p_large, d_result);
  checkCudaErrors(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaDeviceSynchronize());
  if(h_result != expected_result) {
    std::cout << "Test failed" << std::endl;
	 rc=-1;
	 goto Exit;    
  }

  // benchmark default kernel parameter limit
  {
    auto start = steady_clock::now();
    for (int i = 0; i < TEST_ITERATIONS; ++i) {
      checkCudaErrors(cudaMemcpyToSymbol(excess_params, copied_params, CONST_COPIED_PARAMS * sizeof(int), 0, cudaMemcpyHostToDevice));
      kernelDefault<<<1, 1>>>(p, d_result);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = steady_clock::now();
    std::cout << "Kernel 4KB parameter limit - time (us):";
    report_time(start, end, TEST_ITERATIONS);

    // benchmark large kernel parameter limit
    start = steady_clock::now();
    for (int i = 0; i < TEST_ITERATIONS; ++i) {
      kernelLargeParam<<<1, 1>>>(p_large, d_result);
    }  
    checkCudaErrors(cudaDeviceSynchronize());
    end = steady_clock::now();
    std::cout << "Kernel 32,764 byte parameter limit - time (us):";
    report_time(start, end, TEST_ITERATIONS);
  }
  std::cout << "Test passed!" << std::endl;
  rc=0;
Exit:
  // cleanup
  cudaFree(d_result);
  free(copied_params);
  return rc;
}
