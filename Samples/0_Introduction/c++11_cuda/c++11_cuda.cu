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

#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <helper_cuda.h>

/////////////////////////////////////////////////////////////////
// Some utility code to define grid_stride_range
// Normally this would be in a header but it's here
// for didactic purposes. Uses
#include "range.hpp"
using namespace util::lang;

// type alias to simplify typing...
template <typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

template <typename T>
__device__ step_range<T> grid_stride_range(T begin, T end) {
  begin += blockDim.x * blockIdx.x + threadIdx.x;
  return range(begin, end).step(gridDim.x * blockDim.x);
}
/////////////////////////////////////////////////////////////////

template <typename T, typename Predicate>
__device__ void count_if(int *count, T *data, int n, Predicate p) {
  for (auto i : grid_stride_range(0, n)) {
    if (p(data[i])) atomicAdd(count, 1);
  }
}

// Use count_if with a lambda function that searches for x, y, z or w
// Note the use of range-based for loop and initializer_list inside the functor
// We use auto so we don't have to know the type of the functor or array
__global__ void xyzw_frequency(int *count, char *text, int n) {
  const char letters[]{'x', 'y', 'z', 'w'};

  count_if(count, text, n, [&](char c) {
    for (const auto x : letters)
      if (c == x) return true;
    return false;
  });
}

__global__ void xyzw_frequency_thrust_device(int *count, char *text, int n) {
  const char letters[]{'x', 'y', 'z', 'w'};
  *count = thrust::count_if(thrust::device, text, text + n, [=](char c) {
    for (const auto x : letters)
      if (c == x) return true;
    return false;
  });
}

// a bug in Thrust 1.8 causes warnings when this is uncommented
// so commented out by default -- fixed in Thrust master branch
#if 0 
void xyzw_frequency_thrust_host(int *count, char *text, int n)
{
  const char letters[] { 'x','y','z','w' };
  *count = thrust::count_if(thrust::host, text, text+n, [&](char c) {
    for (const auto x : letters) 
      if (c == x) return true;
    return false;
  });
}
#endif

int main(int argc, char **argv) {
  const char *filename = sdkFindFilePath("warandpeace.txt", argv[0]);

  int numBytes = 16 * 1048576;
  char *h_text = (char *)malloc(numBytes);

  // find first CUDA device
  int devID = findCudaDevice(argc, (const char **)argv);

  char *d_text;
  checkCudaErrors(cudaMalloc((void **)&d_text, numBytes));

  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Cannot find the input text file\n. Exiting..\n");
    return EXIT_FAILURE;
  }
  int len = (int)fread(h_text, sizeof(char), numBytes, fp);
  fclose(fp);
  std::cout << "Read " << len << " byte corpus from " << filename << std::endl;

  checkCudaErrors(cudaMemcpy(d_text, h_text, len, cudaMemcpyHostToDevice));

  int count = 0;
  int *d_count;
  checkCudaErrors(cudaMalloc(&d_count, sizeof(int)));
  checkCudaErrors(cudaMemset(d_count, 0, sizeof(int)));

  // Try uncommenting one kernel call at a time
  xyzw_frequency<<<8, 256>>>(d_count, d_text, len);
  xyzw_frequency_thrust_device<<<1, 1>>>(d_count, d_text, len);
  checkCudaErrors(
      cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

  // xyzw_frequency_thrust_host(&count, h_text, len);

  std::cout << "counted " << count
            << " instances of 'x', 'y', 'z', or 'w' in \"" << filename << "\""
            << std::endl;

  checkCudaErrors(cudaFree(d_count));
  checkCudaErrors(cudaFree(d_text));

  return EXIT_SUCCESS;
}
