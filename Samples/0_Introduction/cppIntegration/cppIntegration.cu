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
 * Example of integrating CUDA functions into an existing
 * application / framework.
 * Host part of the device code.
 * Compiled with Cuda compiler.
 */

// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

////////////////////////////////////////////////////////////////////////////////
// declaration, forward

extern "C" void computeGold(char *reference, char *idata,
                            const unsigned int len);
extern "C" void computeGold2(int2 *reference, int2 *idata,
                             const unsigned int len);

///////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_odata  memory to process (in and out)
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel(int *g_data) {
  // write data to global memory
  const unsigned int tid = threadIdx.x;
  int data = g_data[tid];

  // use integer arithmetic to process all four bytes with one thread
  // this serializes the execution, but is the simplest solutions to avoid
  // bank conflicts for this very low number of threads
  // in general it is more efficient to process each byte by a separate thread,
  // to avoid bank conflicts the access pattern should be
  // g_data[4 * wtid + wid], where wtid is the thread id within the half warp
  // and wid is the warp id
  // see also the programming guide for a more in depth discussion.
  g_data[tid] =
      ((((data << 0) >> 24) - 10) << 24) | ((((data << 8) >> 24) - 10) << 16) |
      ((((data << 16) >> 24) - 10) << 8) | ((((data << 24) >> 24) - 10) << 0);
}

///////////////////////////////////////////////////////////////////////////////
//! Demonstration that int2 data can be used in the cpp code
//! @param g_odata  memory to process (in and out)
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel2(int2 *g_data) {
  // write data to global memory
  const unsigned int tid = threadIdx.x;
  int2 data = g_data[tid];

  // use integer arithmetic to process all four bytes with one thread
  // this serializes the execution, but is the simplest solutions to avoid
  // bank conflicts for this very low number of threads
  // in general it is more efficient to process each byte by a separate thread,
  // to avoid bank conflicts the access pattern should be
  // g_data[4 * wtid + wid], where wtid is the thread id within the half warp
  // and wid is the warp id
  // see also the programming guide for a more in depth discussion.
  g_data[tid].x = data.x - data.y;
}

////////////////////////////////////////////////////////////////////////////////
//! Entry point for Cuda functionality on host side
//! @param argc  command line argument count
//! @param argv  command line arguments
//! @param data  data to process on the device
//! @param len   len of \a data
////////////////////////////////////////////////////////////////////////////////
extern "C" bool runTest(const int argc, const char **argv, char *data,
                        int2 *data_int2, unsigned int len) {
  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  findCudaDevice(argc, (const char **)argv);

  const unsigned int num_threads = len / 4;
  assert(0 == (len % 4));
  const unsigned int mem_size = sizeof(char) * len;
  const unsigned int mem_size_int2 = sizeof(int2) * len;

  // allocate device memory
  char *d_data;
  checkCudaErrors(cudaMalloc((void **)&d_data, mem_size));
  // copy host memory to device
  checkCudaErrors(cudaMemcpy(d_data, data, mem_size, cudaMemcpyHostToDevice));
  // allocate device memory for int2 version
  int2 *d_data_int2;
  checkCudaErrors(cudaMalloc((void **)&d_data_int2, mem_size_int2));
  // copy host memory to device
  checkCudaErrors(cudaMemcpy(d_data_int2, data_int2, mem_size_int2,
                             cudaMemcpyHostToDevice));

  // setup execution parameters
  dim3 grid(1, 1, 1);
  dim3 threads(num_threads, 1, 1);
  dim3 threads2(len, 1, 1);  // more threads needed fir separate int2 version
  // execute the kernel
  kernel<<<grid, threads>>>((int *)d_data);
  kernel2<<<grid, threads2>>>(d_data_int2);

  // check if kernel execution generated and error
  getLastCudaError("Kernel execution failed");

  // compute reference solutions
  char *reference = (char *)malloc(mem_size);
  computeGold(reference, data, len);
  int2 *reference2 = (int2 *)malloc(mem_size_int2);
  computeGold2(reference2, data_int2, len);

  // copy results from device to host
  checkCudaErrors(cudaMemcpy(data, d_data, mem_size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(data_int2, d_data_int2, mem_size_int2,
                             cudaMemcpyDeviceToHost));

  // check result
  bool success = true;

  for (unsigned int i = 0; i < len; i++) {
    if (reference[i] != data[i] || reference2[i].x != data_int2[i].x ||
        reference2[i].y != data_int2[i].y) {
      success = false;
    }
  }

  // cleanup memory
  checkCudaErrors(cudaFree(d_data));
  checkCudaErrors(cudaFree(d_data_int2));
  free(reference);
  free(reference2);

  return success;
}
