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
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE
 *
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

// Utilities and system includes
#include <helper_cuda.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples

const char *sSDKname = "conjugateGradient";

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz) {
  I[0] = 0, J[0] = 0, J[1] = 1;
  val[0] = (float)rand() / RAND_MAX + 10.0f;
  val[1] = (float)rand() / RAND_MAX;
  int start;

  for (int i = 1; i < N; i++) {
    if (i > 1) {
      I[i] = I[i - 1] + 3;
    } else {
      I[1] = 2;
    }

    start = (i - 1) * 3 + 2;
    J[start] = i - 1;
    J[start + 1] = i;

    if (i < N - 1) {
      J[start + 2] = i + 1;
    }

    val[start] = val[start - 1];
    val[start + 1] = (float)rand() / RAND_MAX + 10.0f;

    if (i < N - 1) {
      val[start + 2] = (float)rand() / RAND_MAX;
    }
  }

  I[N] = nz;
}

int main(int argc, char **argv) {
  int M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
  float *val = NULL;
  const float tol = 1e-5f;
  const int max_iter = 10000;
  float *x;
  float *rhs;
  float a, b, na, r0, r1;
  int *d_col, *d_row;
  float *d_val, *d_x, dot;
  float *d_r, *d_p, *d_Ax;
  int k;
  float alpha, beta, alpham1;

  // This will pick the best possible CUDA capable device
  cudaDeviceProp deviceProp;
  int devID = findCudaDevice(argc, (const char **)argv);

  if (devID < 0) {
    printf("exiting...\n");
    exit(EXIT_SUCCESS);
  }

  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

  // Statistics about the GPU device
  printf(
      "> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
      deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

  /* Generate a random tridiagonal symmetric matrix in CSR format */
  M = N = 1048576;
  nz = (N - 2) * 3 + 4;
  I = (int *)malloc(sizeof(int) * (N + 1));
  J = (int *)malloc(sizeof(int) * nz);
  val = (float *)malloc(sizeof(float) * nz);
  genTridiag(I, J, val, N, nz);

  x = (float *)malloc(sizeof(float) * N);
  rhs = (float *)malloc(sizeof(float) * N);

  for (int i = 0; i < N; i++) {
    rhs[i] = 1.0;
    x[i] = 0.0;
  }

  /* Get handle to the CUBLAS context */
  cublasHandle_t cublasHandle = 0;
  cublasStatus_t cublasStatus;
  cublasStatus = cublasCreate(&cublasHandle);

  checkCudaErrors(cublasStatus);

  /* Get handle to the CUSPARSE context */
  cusparseHandle_t cusparseHandle = 0;
  checkCudaErrors(cusparseCreate(&cusparseHandle));

  checkCudaErrors(cudaMalloc((void **)&d_col, nz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_row, (N + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_val, nz * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_x, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_Ax, N * sizeof(float)));

  /* Wrap raw data into cuSPARSE generic API objects */
  cusparseSpMatDescr_t matA = NULL;
  checkCudaErrors(cusparseCreateCsr(&matA, N, N, nz, d_row, d_col, d_val,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  cusparseDnVecDescr_t vecx = NULL;
  checkCudaErrors(cusparseCreateDnVec(&vecx, N, d_x, CUDA_R_32F));
  cusparseDnVecDescr_t vecp = NULL;
  checkCudaErrors(cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_32F));
  cusparseDnVecDescr_t vecAx = NULL;
  checkCudaErrors(cusparseCreateDnVec(&vecAx, N, d_Ax, CUDA_R_32F));

  /* Initialize problem data */
  cudaMemcpy(d_col, J, nz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_row, I, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, val, nz * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_r, rhs, N * sizeof(float), cudaMemcpyHostToDevice);

  alpha = 1.0;
  alpham1 = -1.0;
  beta = 0.0;
  r0 = 0.;

  /* Allocate workspace for cuSPARSE */
  size_t bufferSize = 0;
  checkCudaErrors(cusparseSpMV_bufferSize(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
      &beta, vecAx, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  void *buffer = NULL;
  checkCudaErrors(cudaMalloc(&buffer, bufferSize));

  /* Begin CG */
  checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, CUDA_R_32F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

  cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
  cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

  k = 1;

  while (r1 > tol * tol && k <= max_iter) {
    if (k > 1) {
      b = r1 / r0;
      cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
      cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
    } else {
      cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
    }

    checkCudaErrors(cusparseSpMV(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
        &beta, vecAx, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
    a = r1 / dot;

    cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
    na = -a;
    cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

    r0 = r1;
    cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
    cudaDeviceSynchronize();
    printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    k++;
  }

  cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

  float rsum, diff, err = 0.0;

  for (int i = 0; i < N; i++) {
    rsum = 0.0;

    for (int j = I[i]; j < I[i + 1]; j++) {
      rsum += val[j] * x[J[j]];
    }

    diff = fabs(rsum - rhs[i]);

    if (diff > err) {
      err = diff;
    }
  }

  cusparseDestroy(cusparseHandle);
  cublasDestroy(cublasHandle);
  if (matA) {
    checkCudaErrors(cusparseDestroySpMat(matA));
  }
  if (vecx) {
    checkCudaErrors(cusparseDestroyDnVec(vecx));
  }
  if (vecAx) {
    checkCudaErrors(cusparseDestroyDnVec(vecAx));
  }
  if (vecp) {
    checkCudaErrors(cusparseDestroyDnVec(vecp));
  }

  free(I);
  free(J);
  free(val);
  free(x);
  free(rhs);
  cudaFree(d_col);
  cudaFree(d_row);
  cudaFree(d_val);
  cudaFree(d_x);
  cudaFree(d_r);
  cudaFree(d_p);
  cudaFree(d_Ax);

  printf("Test Summary:  Error amount = %f\n", err);
  exit((k <= max_iter) ? 0 : 1);
}
