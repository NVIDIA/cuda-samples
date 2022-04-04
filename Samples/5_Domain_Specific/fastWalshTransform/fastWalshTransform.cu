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
 * Walsh transforms belong to a class of generalized Fourier transformations.
 * They have applications in various fields of electrical engineering
 * and numeric theory. In this sample we demonstrate efficient implementation
 * of naturally-ordered Walsh transform
 * (also known as Walsh-Hadamard or Hadamard transform) in CUDA and its
 * particular application to dyadic convolution computation.
 * Refer to excellent Jorg Arndt's "Algorithms for Programmers" textbook
 * http://www.jjj.de/fxt/fxtbook.pdf (Chapter 22)
 *
 * Victor Podlozhnyuk (vpodlozhnyuk@nvidia.com)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_functions.h>
#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// Reference CPU FWT
////////////////////////////////////////////////////////////////////////////////
extern "C" void fwtCPU(float *h_Output, float *h_Input, int log2N);
extern "C" void slowWTcpu(float *h_Output, float *h_Input, int log2N);
extern "C" void dyadicConvolutionCPU(float *h_Result, float *h_Data,
                                     float *h_Kernel, int log2dataN,
                                     int log2kernelN);

////////////////////////////////////////////////////////////////////////////////
// GPU FWT
////////////////////////////////////////////////////////////////////////////////
#include "fastWalshTransform_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int log2Kernel = 7;
const int log2Data = 23;

const int dataN = 1 << log2Data;
const int kernelN = 1 << log2Kernel;

const int DATA_SIZE = dataN * sizeof(float);
const int KERNEL_SIZE = kernelN * sizeof(float);

const double NOPS = 3.0 * (double)dataN * (double)log2Data / 2.0;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  float *h_Data, *h_Kernel, *h_ResultCPU, *h_ResultGPU;

  float *d_Data, *d_Kernel;

  double delta, ref, sum_delta2, sum_ref2, L2norm, gpuTime;

  StopWatchInterface *hTimer = NULL;
  int i;

  printf("%s Starting...\n\n", argv[0]);

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  findCudaDevice(argc, (const char **)argv);

  sdkCreateTimer(&hTimer);

  printf("Initializing data...\n");
  printf("...allocating CPU memory\n");
  h_Kernel = (float *)malloc(KERNEL_SIZE);
  h_Data = (float *)malloc(DATA_SIZE);
  h_ResultCPU = (float *)malloc(DATA_SIZE);
  h_ResultGPU = (float *)malloc(DATA_SIZE);
  printf("...allocating GPU memory\n");
  checkCudaErrors(cudaMalloc((void **)&d_Kernel, DATA_SIZE));
  checkCudaErrors(cudaMalloc((void **)&d_Data, DATA_SIZE));

  printf("...generating data\n");
  printf("Data length: %i; kernel length: %i\n", dataN, kernelN);
  srand(2007);

  for (i = 0; i < kernelN; i++) {
    h_Kernel[i] = (float)rand() / (float)RAND_MAX;
  }

  for (i = 0; i < dataN; i++) {
    h_Data[i] = (float)rand() / (float)RAND_MAX;
  }

  checkCudaErrors(cudaMemset(d_Kernel, 0, DATA_SIZE));
  checkCudaErrors(
      cudaMemcpy(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_Data, h_Data, DATA_SIZE, cudaMemcpyHostToDevice));

  printf("Running GPU dyadic convolution using Fast Walsh Transform...\n");
  checkCudaErrors(cudaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  fwtBatchGPU(d_Data, 1, log2Data);
  fwtBatchGPU(d_Kernel, 1, log2Data);
  modulateGPU(d_Data, d_Kernel, dataN);
  fwtBatchGPU(d_Data, 1, log2Data);
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  gpuTime = sdkGetTimerValue(&hTimer);
  printf("GPU time: %f ms; GOP/s: %f\n", gpuTime,
         NOPS / (gpuTime * 0.001 * 1E+9));

  printf("Reading back GPU results...\n");
  checkCudaErrors(
      cudaMemcpy(h_ResultGPU, d_Data, DATA_SIZE, cudaMemcpyDeviceToHost));

  printf("Running straightforward CPU dyadic convolution...\n");
  dyadicConvolutionCPU(h_ResultCPU, h_Data, h_Kernel, log2Data, log2Kernel);

  printf("Comparing the results...\n");
  sum_delta2 = 0;
  sum_ref2 = 0;

  for (i = 0; i < dataN; i++) {
    delta = h_ResultCPU[i] - h_ResultGPU[i];
    ref = h_ResultCPU[i];
    sum_delta2 += delta * delta;
    sum_ref2 += ref * ref;
  }

  L2norm = sqrt(sum_delta2 / sum_ref2);

  printf("Shutting down...\n");
  sdkDeleteTimer(&hTimer);
  checkCudaErrors(cudaFree(d_Data));
  checkCudaErrors(cudaFree(d_Kernel));
  free(h_ResultGPU);
  free(h_ResultCPU);
  free(h_Data);
  free(h_Kernel);

  printf("L2 norm: %E\n", L2norm);
  printf(L2norm < 1e-6 ? "Test passed\n" : "Test failed!\n");
}
