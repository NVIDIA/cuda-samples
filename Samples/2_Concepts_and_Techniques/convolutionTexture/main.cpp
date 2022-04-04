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
 * This sample implements the same algorithm as the convolutionSeparable
 * CUDA Sample, but without using the shared memory at all.
 * Instead, it uses textures in exactly the same way an OpenGL-based
 * implementation would do.
 * Refer to the "Performance" section of convolutionSeparable whitepaper.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include "convolutionTexture_common.h"

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  float *h_Kernel, *h_Input, *h_Buffer, *h_OutputCPU, *h_OutputGPU;

  cudaArray *a_Src;
  cudaTextureObject_t texSrc;
  cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

  float *d_Output;

  float gpuTime;

  StopWatchInterface *hTimer = NULL;

  const int imageW = 3072;
  const int imageH = 3072 / 2;
  const unsigned int iterations = 10;

  printf("[%s] - Starting...\n", argv[0]);

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  findCudaDevice(argc, (const char **)argv);

  sdkCreateTimer(&hTimer);

  printf("Initializing data...\n");
  h_Kernel = (float *)malloc(KERNEL_LENGTH * sizeof(float));
  h_Input = (float *)malloc(imageW * imageH * sizeof(float));
  h_Buffer = (float *)malloc(imageW * imageH * sizeof(float));
  h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
  h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
  checkCudaErrors(cudaMallocArray(&a_Src, &floatTex, imageW, imageH));
  checkCudaErrors(
      cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));

  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = a_Src;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.addressMode[1] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(cudaCreateTextureObject(&texSrc, &texRes, &texDescr, NULL));

  srand(2009);

  for (unsigned int i = 0; i < KERNEL_LENGTH; i++) {
    h_Kernel[i] = (float)(rand() % 16);
  }

  for (unsigned int i = 0; i < imageW * imageH; i++) {
    h_Input[i] = (float)(rand() % 16);
  }

  setConvolutionKernel(h_Kernel);
  checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, h_Input,
                                    imageW * imageH * sizeof(float),
                                    cudaMemcpyHostToDevice));

  printf("Running GPU rows convolution (%u identical iterations)...\n",
         iterations);
  checkCudaErrors(cudaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  for (unsigned int i = 0; i < iterations; i++) {
    convolutionRowsGPU(d_Output, a_Src, imageW, imageH, texSrc);
  }

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
  printf("Average convolutionRowsGPU() time: %f msecs; //%f Mpix/s\n", gpuTime,
         imageW * imageH * 1e-6 / (0.001 * gpuTime));

  // While CUDA kernels can't write to textures directly, this copy is
  // inevitable
  printf("Copying convolutionRowGPU() output back to the texture...\n");
  checkCudaErrors(cudaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, d_Output,
                                    imageW * imageH * sizeof(float),
                                    cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  gpuTime = sdkGetTimerValue(&hTimer);
  printf("cudaMemcpyToArray() time: %f msecs; //%f Mpix/s\n", gpuTime,
         imageW * imageH * 1e-6 / (0.001 * gpuTime));

  printf("Running GPU columns convolution (%i iterations)\n", iterations);
  checkCudaErrors(cudaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  for (int i = 0; i < iterations; i++) {
    convolutionColumnsGPU(d_Output, a_Src, imageW, imageH, texSrc);
  }

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
  printf("Average convolutionColumnsGPU() time: %f msecs; //%f Mpix/s\n",
         gpuTime, imageW * imageH * 1e-6 / (0.001 * gpuTime));

  printf("Reading back GPU results...\n");
  checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output,
                             imageW * imageH * sizeof(float),
                             cudaMemcpyDeviceToHost));

  printf("Checking the results...\n");
  printf("...running convolutionRowsCPU()\n");
  convolutionRowsCPU(h_Buffer, h_Input, h_Kernel, imageW, imageH,
                     KERNEL_RADIUS);

  printf("...running convolutionColumnsCPU()\n");
  convolutionColumnsCPU(h_OutputCPU, h_Buffer, h_Kernel, imageW, imageH,
                        KERNEL_RADIUS);

  double delta = 0;
  double sum = 0;

  for (unsigned int i = 0; i < imageW * imageH; i++) {
    sum += h_OutputCPU[i] * h_OutputCPU[i];
    delta +=
        (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
  }

  double L2norm = sqrt(delta / sum);
  printf("Relative L2 norm: %E\n", L2norm);
  printf("Shutting down...\n");

  checkCudaErrors(cudaFree(d_Output));
  checkCudaErrors(cudaFreeArray(a_Src));
  free(h_OutputGPU);
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
