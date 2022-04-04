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
 * This sample demonstrates how 2D convolutions
 * with very large kernel sizes
 * can be efficiently implemented
 * using FFT transformations.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "convolutionFFT2D_common.h"

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
int snapTransformSize(int dataSize) {
  int hiBit;
  unsigned int lowPOT, hiPOT;

  dataSize = iAlignUp(dataSize, 16);

  for (hiBit = 31; hiBit >= 0; hiBit--)
    if (dataSize & (1U << hiBit)) {
      break;
    }

  lowPOT = 1U << hiBit;

  if (lowPOT == (unsigned int)dataSize) {
    return dataSize;
  }

  hiPOT = 1U << (hiBit + 1);

  if (hiPOT <= 1024) {
    return hiPOT;
  } else {
    return iAlignUp(dataSize, 512);
  }
}

float getRand(void) { return (float)(rand() % 16); }

bool test0(void) {
  float *h_Data, *h_Kernel, *h_ResultCPU, *h_ResultGPU;

  float *d_Data, *d_PaddedData, *d_Kernel, *d_PaddedKernel;

  fComplex *d_DataSpectrum, *d_KernelSpectrum;

  cufftHandle fftPlanFwd, fftPlanInv;

  bool bRetVal;
  StopWatchInterface *hTimer = NULL;
  sdkCreateTimer(&hTimer);

  printf("Testing built-in R2C / C2R FFT-based convolution\n");
  const int kernelH = 7;
  const int kernelW = 6;
  const int kernelY = 3;
  const int kernelX = 4;
  const int dataH = 2000;
  const int dataW = 2000;
  const int fftH = snapTransformSize(dataH + kernelH - 1);
  const int fftW = snapTransformSize(dataW + kernelW - 1);

  printf("...allocating memory\n");
  h_Data = (float *)malloc(dataH * dataW * sizeof(float));
  h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
  h_ResultCPU = (float *)malloc(dataH * dataW * sizeof(float));
  h_ResultGPU = (float *)malloc(fftH * fftW * sizeof(float));

  checkCudaErrors(cudaMalloc((void **)&d_Data, dataH * dataW * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));

  checkCudaErrors(
      cudaMalloc((void **)&d_PaddedData, fftH * fftW * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

  checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum,
                             fftH * (fftW / 2 + 1) * sizeof(fComplex)));
  checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum,
                             fftH * (fftW / 2 + 1) * sizeof(fComplex)));
  checkCudaErrors(cudaMemset(d_KernelSpectrum, 0,
                             fftH * (fftW / 2 + 1) * sizeof(fComplex)));

  printf("...generating random input data\n");
  srand(2010);

  for (int i = 0; i < dataH * dataW; i++) {
    h_Data[i] = getRand();
  }

  for (int i = 0; i < kernelH * kernelW; i++) {
    h_Kernel[i] = getRand();
  }

  printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
  checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
  checkCudaErrors(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));

  printf("...uploading to GPU and padding convolution kernel and input data\n");
  checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel,
                             kernelH * kernelW * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Data, h_Data, dataH * dataW * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
  checkCudaErrors(cudaMemset(d_PaddedData, 0, fftH * fftW * sizeof(float)));

  padKernel(d_PaddedKernel, d_Kernel, fftH, fftW, kernelH, kernelW, kernelY,
            kernelX);

  padDataClampToBorder(d_PaddedData, d_Data, fftH, fftW, dataH, dataW, kernelH,
                       kernelW, kernelY, kernelX);

  // Not including kernel transformation into time measurement,
  // since convolution kernel is not changed very frequently
  printf("...transforming convolution kernel\n");
  checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedKernel,
                               (cufftComplex *)d_KernelSpectrum));

  printf("...running GPU FFT convolution: ");
  checkCudaErrors(cudaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedData,
                               (cufftComplex *)d_DataSpectrum));
  modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
  checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_DataSpectrum,
                               (cufftReal *)d_PaddedData));

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  double gpuTime = sdkGetTimerValue(&hTimer);
  printf("%f MPix/s (%f ms)\n",
         (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

  printf("...reading back GPU convolution results\n");
  checkCudaErrors(cudaMemcpy(h_ResultGPU, d_PaddedData,
                             fftH * fftW * sizeof(float),
                             cudaMemcpyDeviceToHost));

  printf("...running reference CPU convolution\n");
  convolutionClampToBorderCPU(h_ResultCPU, h_Data, h_Kernel, dataH, dataW,
                              kernelH, kernelW, kernelY, kernelX);

  printf("...comparing the results: ");
  double sum_delta2 = 0;
  double sum_ref2 = 0;
  double max_delta_ref = 0;

  for (int y = 0; y < dataH; y++)
    for (int x = 0; x < dataW; x++) {
      double rCPU = (double)h_ResultCPU[y * dataW + x];
      double rGPU = (double)h_ResultGPU[y * fftW + x];
      double delta = (rCPU - rGPU) * (rCPU - rGPU);
      double ref = rCPU * rCPU + rCPU * rCPU;

      if ((delta / ref) > max_delta_ref) {
        max_delta_ref = delta / ref;
      }

      sum_delta2 += delta;
      sum_ref2 += ref;
    }

  double L2norm = sqrt(sum_delta2 / sum_ref2);
  printf("rel L2 = %E (max delta = %E)\n", L2norm, sqrt(max_delta_ref));
  bRetVal = (L2norm < 1e-6) ? true : false;
  printf(bRetVal ? "L2norm Error OK\n" : "L2norm Error too high!\n");

  printf("...shutting down\n");
  sdkDeleteTimer(&hTimer);

  checkCudaErrors(cufftDestroy(fftPlanInv));
  checkCudaErrors(cufftDestroy(fftPlanFwd));

  checkCudaErrors(cudaFree(d_DataSpectrum));
  checkCudaErrors(cudaFree(d_KernelSpectrum));
  checkCudaErrors(cudaFree(d_PaddedData));
  checkCudaErrors(cudaFree(d_PaddedKernel));
  checkCudaErrors(cudaFree(d_Data));
  checkCudaErrors(cudaFree(d_Kernel));

  free(h_ResultGPU);
  free(h_ResultCPU);
  free(h_Data);
  free(h_Kernel);

  return bRetVal;
}

bool test1(void) {
  float *h_Data, *h_Kernel, *h_ResultCPU, *h_ResultGPU;

  float *d_Data, *d_Kernel, *d_PaddedData, *d_PaddedKernel;

  fComplex *d_DataSpectrum0, *d_KernelSpectrum0, *d_DataSpectrum,
      *d_KernelSpectrum;

  cufftHandle fftPlan;

  bool bRetVal;
  StopWatchInterface *hTimer = NULL;
  sdkCreateTimer(&hTimer);

  printf("Testing custom R2C / C2R FFT-based convolution\n");
  const uint fftPadding = 16;
  const int kernelH = 7;
  const int kernelW = 6;
  const int kernelY = 3;
  const int kernelX = 4;
  const int dataH = 2000;
  const int dataW = 2000;
  const int fftH = snapTransformSize(dataH + kernelH - 1);
  const int fftW = snapTransformSize(dataW + kernelW - 1);

  printf("...allocating memory\n");
  h_Data = (float *)malloc(dataH * dataW * sizeof(float));
  h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
  h_ResultCPU = (float *)malloc(dataH * dataW * sizeof(float));
  h_ResultGPU = (float *)malloc(fftH * fftW * sizeof(float));

  checkCudaErrors(cudaMalloc((void **)&d_Data, dataH * dataW * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));

  checkCudaErrors(
      cudaMalloc((void **)&d_PaddedData, fftH * fftW * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

  checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum0,
                             fftH * (fftW / 2) * sizeof(fComplex)));
  checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum0,
                             fftH * (fftW / 2) * sizeof(fComplex)));
  checkCudaErrors(
      cudaMalloc((void **)&d_DataSpectrum,
                 fftH * (fftW / 2 + fftPadding) * sizeof(fComplex)));
  checkCudaErrors(
      cudaMalloc((void **)&d_KernelSpectrum,
                 fftH * (fftW / 2 + fftPadding) * sizeof(fComplex)));

  printf("...generating random input data\n");
  srand(2010);

  for (int i = 0; i < dataH * dataW; i++) {
    h_Data[i] = getRand();
  }

  for (int i = 0; i < kernelH * kernelW; i++) {
    h_Kernel[i] = getRand();
  }

  printf("...creating C2C FFT plan for %i x %i\n", fftH, fftW / 2);
  checkCudaErrors(cufftPlan2d(&fftPlan, fftH, fftW / 2, CUFFT_C2C));

  printf("...uploading to GPU and padding convolution kernel and input data\n");
  checkCudaErrors(cudaMemcpy(d_Data, h_Data, dataH * dataW * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel,
                             kernelH * kernelW * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_PaddedData, 0, fftH * fftW * sizeof(float)));
  checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));

  padDataClampToBorder(d_PaddedData, d_Data, fftH, fftW, dataH, dataW, kernelH,
                       kernelW, kernelY, kernelX);

  padKernel(d_PaddedKernel, d_Kernel, fftH, fftW, kernelH, kernelW, kernelY,
            kernelX);

  // CUFFT_INVERSE works just as well...
  const int FFT_DIR = CUFFT_FORWARD;

  // Not including kernel transformation into time measurement,
  // since convolution kernel is not changed very frequently
  printf("...transforming convolution kernel\n");
  checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_PaddedKernel,
                               (cufftComplex *)d_KernelSpectrum0, FFT_DIR));
  spPostprocess2D(d_KernelSpectrum, d_KernelSpectrum0, fftH, fftW / 2,
                  fftPadding, FFT_DIR);

  printf("...running GPU FFT convolution: ");
  checkCudaErrors(cudaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_PaddedData,
                               (cufftComplex *)d_DataSpectrum0, FFT_DIR));

  spPostprocess2D(d_DataSpectrum, d_DataSpectrum0, fftH, fftW / 2, fftPadding,
                  FFT_DIR);
  modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW,
                       fftPadding);
  spPreprocess2D(d_DataSpectrum0, d_DataSpectrum, fftH, fftW / 2, fftPadding,
                 -FFT_DIR);

  checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_DataSpectrum0,
                               (cufftComplex *)d_PaddedData, -FFT_DIR));

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  double gpuTime = sdkGetTimerValue(&hTimer);
  printf("%f MPix/s (%f ms)\n",
         (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

  printf("...reading back GPU FFT results\n");
  checkCudaErrors(cudaMemcpy(h_ResultGPU, d_PaddedData,
                             fftH * fftW * sizeof(float),
                             cudaMemcpyDeviceToHost));

  printf("...running reference CPU convolution\n");
  convolutionClampToBorderCPU(h_ResultCPU, h_Data, h_Kernel, dataH, dataW,
                              kernelH, kernelW, kernelY, kernelX);

  printf("...comparing the results: ");
  double sum_delta2 = 0;
  double sum_ref2 = 0;
  double max_delta_ref = 0;

  for (int y = 0; y < dataH; y++)
    for (int x = 0; x < dataW; x++) {
      double rCPU = (double)h_ResultCPU[y * dataW + x];
      double rGPU = (double)h_ResultGPU[y * fftW + x];
      double delta = (rCPU - rGPU) * (rCPU - rGPU);
      double ref = rCPU * rCPU + rCPU * rCPU;

      if ((delta / ref) > max_delta_ref) {
        max_delta_ref = delta / ref;
      }

      sum_delta2 += delta;
      sum_ref2 += ref;
    }

  double L2norm = sqrt(sum_delta2 / sum_ref2);
  printf("rel L2 = %E (max delta = %E)\n", L2norm, sqrt(max_delta_ref));
  bRetVal = (L2norm < 1e-6) ? true : false;
  printf(bRetVal ? "L2norm Error OK\n" : "L2norm Error too high!\n");

  printf("...shutting down\n");
  sdkDeleteTimer(&hTimer);
  checkCudaErrors(cufftDestroy(fftPlan));

  checkCudaErrors(cudaFree(d_KernelSpectrum));
  checkCudaErrors(cudaFree(d_DataSpectrum));
  checkCudaErrors(cudaFree(d_KernelSpectrum0));
  checkCudaErrors(cudaFree(d_DataSpectrum0));
  checkCudaErrors(cudaFree(d_PaddedKernel));
  checkCudaErrors(cudaFree(d_PaddedData));
  checkCudaErrors(cudaFree(d_Kernel));
  checkCudaErrors(cudaFree(d_Data));

  free(h_ResultGPU);
  free(h_ResultCPU);
  free(h_Kernel);
  free(h_Data);

  return bRetVal;
}

bool test2(void) {
  float *h_Data, *h_Kernel, *h_ResultCPU, *h_ResultGPU;

  float *d_Data, *d_Kernel, *d_PaddedData, *d_PaddedKernel;

  fComplex *d_DataSpectrum0, *d_KernelSpectrum0;

  cufftHandle fftPlan;

  bool bRetVal;
  StopWatchInterface *hTimer = NULL;
  sdkCreateTimer(&hTimer);

  printf("Testing updated custom R2C / C2R FFT-based convolution\n");
  const int kernelH = 7;
  const int kernelW = 6;
  const int kernelY = 3;
  const int kernelX = 4;
  const int dataH = 2000;
  const int dataW = 2000;
  const int fftH = snapTransformSize(dataH + kernelH - 1);
  const int fftW = snapTransformSize(dataW + kernelW - 1);

  printf("...allocating memory\n");
  h_Data = (float *)malloc(dataH * dataW * sizeof(float));
  h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
  h_ResultCPU = (float *)malloc(dataH * dataW * sizeof(float));
  h_ResultGPU = (float *)malloc(fftH * fftW * sizeof(float));

  checkCudaErrors(cudaMalloc((void **)&d_Data, dataH * dataW * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));

  checkCudaErrors(
      cudaMalloc((void **)&d_PaddedData, fftH * fftW * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

  checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum0,
                             fftH * (fftW / 2) * sizeof(fComplex)));
  checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum0,
                             fftH * (fftW / 2) * sizeof(fComplex)));

  printf("...generating random input data\n");
  srand(2010);

  for (int i = 0; i < dataH * dataW; i++) {
    h_Data[i] = getRand();
  }

  for (int i = 0; i < kernelH * kernelW; i++) {
    h_Kernel[i] = getRand();
  }

  printf("...creating C2C FFT plan for %i x %i\n", fftH, fftW / 2);
  checkCudaErrors(cufftPlan2d(&fftPlan, fftH, fftW / 2, CUFFT_C2C));

  printf("...uploading to GPU and padding convolution kernel and input data\n");
  checkCudaErrors(cudaMemcpy(d_Data, h_Data, dataH * dataW * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel,
                             kernelH * kernelW * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_PaddedData, 0, fftH * fftW * sizeof(float)));
  checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));

  padDataClampToBorder(d_PaddedData, d_Data, fftH, fftW, dataH, dataW, kernelH,
                       kernelW, kernelY, kernelX);

  padKernel(d_PaddedKernel, d_Kernel, fftH, fftW, kernelH, kernelW, kernelY,
            kernelX);

  // CUFFT_INVERSE works just as well...
  const int FFT_DIR = CUFFT_FORWARD;

  // Not including kernel transformation into time measurement,
  // since convolution kernel is not changed very frequently
  printf("...transforming convolution kernel\n");
  checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_PaddedKernel,
                               (cufftComplex *)d_KernelSpectrum0, FFT_DIR));

  printf("...running GPU FFT convolution: ");
  checkCudaErrors(cudaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_PaddedData,
                               (cufftComplex *)d_DataSpectrum0, FFT_DIR));
  spProcess2D(d_DataSpectrum0, d_DataSpectrum0, d_KernelSpectrum0, fftH,
              fftW / 2, FFT_DIR);
  checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_DataSpectrum0,
                               (cufftComplex *)d_PaddedData, -FFT_DIR));

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  double gpuTime = sdkGetTimerValue(&hTimer);
  printf("%f MPix/s (%f ms)\n",
         (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

  printf("...reading back GPU FFT results\n");
  checkCudaErrors(cudaMemcpy(h_ResultGPU, d_PaddedData,
                             fftH * fftW * sizeof(float),
                             cudaMemcpyDeviceToHost));

  printf("...running reference CPU convolution\n");
  convolutionClampToBorderCPU(h_ResultCPU, h_Data, h_Kernel, dataH, dataW,
                              kernelH, kernelW, kernelY, kernelX);

  printf("...comparing the results: ");
  double sum_delta2 = 0;
  double sum_ref2 = 0;
  double max_delta_ref = 0;

  for (int y = 0; y < dataH; y++) {
    for (int x = 0; x < dataW; x++) {
      double rCPU = (double)h_ResultCPU[y * dataW + x];
      double rGPU = (double)h_ResultGPU[y * fftW + x];
      double delta = (rCPU - rGPU) * (rCPU - rGPU);
      double ref = rCPU * rCPU + rCPU * rCPU;

      if ((delta / ref) > max_delta_ref) {
        max_delta_ref = delta / ref;
      }

      sum_delta2 += delta;
      sum_ref2 += ref;
    }
  }

  double L2norm = sqrt(sum_delta2 / sum_ref2);
  printf("rel L2 = %E (max delta = %E)\n", L2norm, sqrt(max_delta_ref));
  bRetVal = (L2norm < 1e-6) ? true : false;
  printf(bRetVal ? "L2norm Error OK\n" : "L2norm Error too high!\n");

  printf("...shutting down\n");
  sdkDeleteTimer(&hTimer);
  checkCudaErrors(cufftDestroy(fftPlan));

  checkCudaErrors(cudaFree(d_KernelSpectrum0));
  checkCudaErrors(cudaFree(d_DataSpectrum0));
  checkCudaErrors(cudaFree(d_PaddedKernel));
  checkCudaErrors(cudaFree(d_PaddedData));
  checkCudaErrors(cudaFree(d_Kernel));
  checkCudaErrors(cudaFree(d_Data));

  free(h_ResultGPU);
  free(h_ResultCPU);
  free(h_Kernel);
  free(h_Data);

  return bRetVal;
}

int main(int argc, char **argv) {
  printf("[%s] - Starting...\n", argv[0]);

  // Use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  findCudaDevice(argc, (const char **)argv);

  int nFailures = 0;

  if (!test0()) {
    nFailures++;
  }

  if (!test1()) {
    nFailures++;
  }

  if (!test2()) {
    nFailures++;
  }

  printf("Test Summary: %d errors\n", nFailures);

  if (nFailures > 0) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
