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

/* Template project which demonstrates the basics on how to setup a project
 * example application.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

extern "C" void computeGold(float *reference, float *idata,
                            const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void testKernel(float *g_idata, float *g_odata) {
  // shared memory
  // the size is determined by the host application
  extern __shared__ float sdata[];

  // access thread id
  const unsigned int tid = threadIdx.x;
  // access number of threads in this block
  const unsigned int num_threads = blockDim.x;

  // read in input data from global memory
  sdata[tid] = g_idata[tid];
  __syncthreads();

  // perform some computations
  sdata[tid] = (float)num_threads * sdata[tid];
  __syncthreads();

  // write data to global memory
  g_odata[tid] = sdata[tid];
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { runTest(argc, argv); }

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  bool bTestResult = true;

  printf("%s Starting...\n\n", argv[0]);

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  int devID = findCudaDevice(argc, (const char **)argv);

  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  unsigned int num_threads = 32;
  unsigned int mem_size = sizeof(float) * num_threads;

  // allocate host memory
  float *h_idata = (float *)malloc(mem_size);

  // initalize the memory
  for (unsigned int i = 0; i < num_threads; ++i) {
    h_idata[i] = (float)i;
  }

  // allocate device memory
  float *d_idata;
  checkCudaErrors(cudaMalloc((void **)&d_idata, mem_size));
  // copy host memory to device
  checkCudaErrors(
      cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

  // allocate device memory for result
  float *d_odata;
  checkCudaErrors(cudaMalloc((void **)&d_odata, mem_size));

  // setup execution parameters
  dim3 grid(1, 1, 1);
  dim3 threads(num_threads, 1, 1);

  // execute the kernel
  testKernel<<<grid, threads, mem_size>>>(d_idata, d_odata);

  // check if kernel execution generated and error
  getLastCudaError("Kernel execution failed");

  // allocate mem for the result on host side
  float *h_odata = (float *)malloc(mem_size);
  // copy result from device to host
  checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * num_threads,
                             cudaMemcpyDeviceToHost));

  sdkStopTimer(&timer);
  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  sdkDeleteTimer(&timer);

  // compute reference solution
  float *reference = (float *)malloc(mem_size);
  computeGold(reference, h_idata, num_threads);

  // check result
  if (checkCmdLineFlag(argc, (const char **)argv, "regression")) {
    // write file for regression test
    sdkWriteFile("./data/regression.dat", h_odata, num_threads, 0.0f, false);
  } else {
    // custom output handling when no regression test running
    // in this case check if the result is equivalent to the expected solution
    bTestResult = compareData(reference, h_odata, num_threads, 0.0f, 0.0f);
  }

  // cleanup memory
  free(h_idata);
  free(h_odata);
  free(reference);
  checkCudaErrors(cudaFree(d_idata));
  checkCudaErrors(cudaFree(d_odata));

  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
