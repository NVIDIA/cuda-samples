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
 * Multi-GPU sample using OpenMP for threading on the CPU side
 * needs a compiler that supports OpenMP 2.0
 */

#include <helper_cuda.h>
#include <omp.h>
#include <stdio.h>  // stdio functions are used since C++ streams aren't necessarily thread safe

using namespace std;

// a simple kernel that simply increments each array element by b
__global__ void kernelAddConstant(int *g_a, const int b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_a[idx] += b;
}

// a predicate that checks whether each array element is set to its index plus b
int correctResult(int *data, const int n, const int b) {
  for (int i = 0; i < n; i++)
    if (data[i] != i + b) return 0;

  return 1;
}

int main(int argc, char *argv[]) {
  int num_gpus = 0;  // number of CUDA GPUs

  printf("%s Starting...\n\n", argv[0]);

  /////////////////////////////////////////////////////////////////
  // determine the number of CUDA capable GPUs
  //
  cudaGetDeviceCount(&num_gpus);

  if (num_gpus < 1) {
    printf("no CUDA capable devices were detected\n");
    return 1;
  }

  /////////////////////////////////////////////////////////////////
  // display CPU and GPU configuration
  //
  printf("number of host CPUs:\t%d\n", omp_get_num_procs());
  printf("number of CUDA devices:\t%d\n", num_gpus);

  for (int i = 0; i < num_gpus; i++) {
    cudaDeviceProp dprop;
    cudaGetDeviceProperties(&dprop, i);
    printf("   %d: %s\n", i, dprop.name);
  }

  printf("---------------------------\n");

  /////////////////////////////////////////////////////////////////
  // initialize data
  //
  unsigned int n = num_gpus * 8192;
  unsigned int nbytes = n * sizeof(int);
  int *a = 0;  // pointer to data on the CPU
  int b = 3;   // value by which the array is incremented
  a = (int *)malloc(nbytes);

  if (0 == a) {
    printf("couldn't allocate CPU memory\n");
    return 1;
  }

  for (unsigned int i = 0; i < n; i++) a[i] = i;

  ////////////////////////////////////////////////////////////////
  // run as many CPU threads as there are CUDA devices
  //   each CPU thread controls a different device, processing its
  //   portion of the data.  It's possible to use more CPU threads
  //   than there are CUDA devices, in which case several CPU
  //   threads will be allocating resources and launching kernels
  //   on the same device.  For example, try omp_set_num_threads(2*num_gpus);
  //   Recall that all variables declared inside an "omp parallel" scope are
  //   local to each CPU thread
  //
  omp_set_num_threads(
      num_gpus);  // create as many CPU threads as there are CUDA devices
// omp_set_num_threads(2*num_gpus);// create twice as many CPU threads as there
// are CUDA devices
#pragma omp parallel
  {
    unsigned int cpu_thread_id = omp_get_thread_num();
    unsigned int num_cpu_threads = omp_get_num_threads();

    // set and check the CUDA device for this CPU thread
    int gpu_id = -1;
    checkCudaErrors(cudaSetDevice(
        cpu_thread_id %
        num_gpus));  // "% num_gpus" allows more CPU threads than GPU devices
    checkCudaErrors(cudaGetDevice(&gpu_id));
    printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id,
           num_cpu_threads, gpu_id);

    int *d_a =
        0;  // pointer to memory on the device associated with this CPU thread
    int *sub_a =
        a +
        cpu_thread_id * n /
            num_cpu_threads;  // pointer to this CPU thread's portion of data
    unsigned int nbytes_per_kernel = nbytes / num_cpu_threads;
    dim3 gpu_threads(128);  // 128 threads per block
    dim3 gpu_blocks(n / (gpu_threads.x * num_cpu_threads));

    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes_per_kernel));
    checkCudaErrors(cudaMemset(d_a, 0, nbytes_per_kernel));
    checkCudaErrors(
        cudaMemcpy(d_a, sub_a, nbytes_per_kernel, cudaMemcpyHostToDevice));
    kernelAddConstant<<<gpu_blocks, gpu_threads>>>(d_a, b);

    checkCudaErrors(
        cudaMemcpy(sub_a, d_a, nbytes_per_kernel, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_a));
  }
  printf("---------------------------\n");

  if (cudaSuccess != cudaGetLastError())
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  ////////////////////////////////////////////////////////////////
  // check the result
  //
  bool bResult = correctResult(a, n, b);

  if (a) free(a);  // free CPU memory

  exit(bResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
