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
* This sample implements a separable convolution filter
* of a 2D image with an arbitrary kernel.
*/

// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include "convolutionSeparable_common.h"

////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowCPU(float *h_Result, float *h_Data,
                                  float *h_Kernel, int imageW, int imageH,
                                  int kernelR);

extern "C" void convolutionColumnCPU(float *h_Result, float *h_Data,
                                     float *h_Kernel, int imageW, int imageH,
                                     int kernelR);

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // start logs
  printf("[%s] - Starting...\n", argv[0]);

  float *h_Kernel, *h_Input, *h_Buffer, *h_OutputCPU, *h_OutputGPU;

  float *d_Input, *d_Output, *d_Buffer;

  const int imageW = 3072;
  const int imageH = 3072;
  const int iterations = 16;

  StopWatchInterface *hTimer = NULL;

  // Use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  findCudaDevice(argc, (const char **)argv);

  sdkCreateTimer(&hTimer);

  printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
  printf("Allocating and initializing host arrays...\n");
  h_Kernel = (float *)malloc(KERNEL_LENGTH * sizeof(float));
  h_Input = (float *)malloc(imageW * imageH * sizeof(float));
  h_Buffer = (float *)malloc(imageW * imageH * sizeof(float));
  h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
  h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
  srand(200);

  for (unsigned int i = 0; i < KERNEL_LENGTH; i++) {
    h_Kernel[i] = (float)(rand() % 16);
  }

  for (unsigned i = 0; i < imageW * imageH; i++) {
    h_Input[i] = (float)(rand() % 16);
  }

  printf("Allocating and initializing CUDA arrays...\n");
  checkCudaErrors(
      cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));

  setConvolutionKernel(h_Kernel);
  checkCudaErrors(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float),
                             cudaMemcpyHostToDevice));

  printf("Running GPU convolution (%u identical iterations)...\n\n",
         iterations);

  for (int i = -1; i < iterations; i++) {
    // i == -1 -- warmup iteration
    if (i == 0) {
      checkCudaErrors(cudaDeviceSynchronize());
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);
    }

    convolutionRowsGPU(d_Buffer, d_Input, imageW, imageH);

    convolutionColumnsGPU(d_Output, d_Buffer, imageW, imageH);
  }

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  double gpuTime = 0.001 * sdkGetTimerValue(&hTimer) / (double)iterations;
  printf(
      "convolutionSeparable, Throughput = %.4f MPixels/sec, Time = %.5f s, "
      "Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n",
      (1.0e-6 * (double)(imageW * imageH) / gpuTime), gpuTime,
      (imageW * imageH), 1, 0);

  printf("\nReading back GPU results...\n\n");
  checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output,
                             imageW * imageH * sizeof(float),
                             cudaMemcpyDeviceToHost));

  printf("Checking the results...\n");
  printf(" ...running convolutionRowCPU()\n");
  convolutionRowCPU(h_Buffer, h_Input, h_Kernel, imageW, imageH, KERNEL_RADIUS);

  printf(" ...running convolutionColumnCPU()\n");
  convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Kernel, imageW, imageH,
                       KERNEL_RADIUS);

  printf(" ...comparing the results\n");
  double sum = 0, delta = 0;

  for (unsigned i = 0; i < imageW * imageH; i++) {
    delta +=
        (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
    sum += h_OutputCPU[i] * h_OutputCPU[i];
  }

  double L2norm = sqrt(delta / sum);
  printf(" ...Relative L2 norm: %E\n\n", L2norm);
  printf("Shutting down...\n");

  checkCudaErrors(cudaFree(d_Buffer));
  checkCudaErrors(cudaFree(d_Output));
  checkCudaErrors(cudaFree(d_Input));
  free(h_OutputGPU);
  free(h_OutputCPU);
  free(h_Buffer);
  free(h_Input);
  free(h_Kernel);

  sdkDeleteTimer(&hTimer);

  if (L2norm > 1e-6) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
