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
 * using CUBLAS and CUSPARSE with CUDA Graphs
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

const char *sSDKname = "conjugateGradientCudaGraphs";

#ifndef WITH_GRAPH
#define WITH_GRAPH 1
#endif

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

__global__ void initVectors(float *rhs, float *x, int N) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = gid; i < N; i += gridDim.x * blockDim.x) {
    rhs[i] = 1.0;
    x[i] = 0.0;
  }
}

__global__ void r1_div_x(float *r1, float *r0, float *b) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid == 0) {
    b[0] = r1[0] / r0[0];
  }
}

__global__ void a_minus(float *a, float *na) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid == 0) {
    na[0] = -(a[0]);
  }
}

int main(int argc, char **argv) {
  int N = 0, nz = 0, *I = NULL, *J = NULL;
  float *val = NULL;
  const float tol = 1e-5f;
  const int max_iter = 10000;
  float *x;
  float *rhs;
  float r1;

  int *d_col, *d_row;
  float *d_val, *d_x;
  float *d_r, *d_p, *d_Ax;
  int k;
  float alpha, beta, alpham1;

  cudaStream_t stream1, streamForGraph;

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
  N = 1048576;
  nz = (N - 2) * 3 + 4;
  checkCudaErrors(cudaMallocHost(&I, sizeof(int) * (N + 1)));
  checkCudaErrors(cudaMallocHost(&J, sizeof(int) * nz));
  checkCudaErrors(cudaMallocHost(&val, sizeof(float) * nz));
  genTridiag(I, J, val, N, nz);

  checkCudaErrors(cudaMallocHost(&x, sizeof(float) * N));
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
  cusparseStatus_t cusparseStatus;
  cusparseStatus = cusparseCreate(&cusparseHandle);

  checkCudaErrors(cusparseStatus);

  checkCudaErrors(cudaStreamCreate(&stream1));

  checkCudaErrors(cudaMalloc((void **)&d_col, nz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_row, (N + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_val, nz * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_x, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_Ax, N * sizeof(float)));

  float *d_r1, *d_r0, *d_dot, *d_a, *d_na, *d_b;
  checkCudaErrors(cudaMalloc((void **)&d_r1, sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_r0, sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_dot, sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_a, sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_na, sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(float)));

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

  /* Allocate workspace for cuSPARSE */
  size_t bufferSize = 0;
  checkCudaErrors(cusparseSpMV_bufferSize(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
      &beta, vecAx, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  void *buffer = NULL;
  checkCudaErrors(cudaMalloc(&buffer, bufferSize));

  cusparseMatDescr_t descr = 0;
  checkCudaErrors(cusparseCreateMatDescr(&descr));

  checkCudaErrors(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  checkCudaErrors(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

  int numBlocks = 0, blockSize = 0;
  checkCudaErrors(
      cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, initVectors));

  checkCudaErrors(cudaMemcpyAsync(d_col, J, nz * sizeof(int),
                                  cudaMemcpyHostToDevice, stream1));
  checkCudaErrors(cudaMemcpyAsync(d_row, I, (N + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice, stream1));
  checkCudaErrors(cudaMemcpyAsync(d_val, val, nz * sizeof(float),
                                  cudaMemcpyHostToDevice, stream1));

  initVectors<<<numBlocks, blockSize, 0, stream1>>>(d_r, d_x, N);

  alpha = 1.0;
  alpham1 = -1.0;
  beta = 0.0;

  checkCudaErrors(cusparseSetStream(cusparseHandle, stream1));
  checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, CUDA_R_32F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

  checkCudaErrors(cublasSetStream(cublasHandle, stream1));
  checkCudaErrors(cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1));

  checkCudaErrors(
      cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE));
  checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, d_r1));

  k = 1;
  // First Iteration when k=1 starts
  checkCudaErrors(cublasScopy(cublasHandle, N, d_r, 1, d_p, 1));
  checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecp, &beta, vecAx, CUDA_R_32F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

  checkCudaErrors(cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, d_dot));

  r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_dot, d_a);

  checkCudaErrors(cublasSaxpy(cublasHandle, N, d_a, d_p, 1, d_x, 1));

  a_minus<<<1, 1, 0, stream1>>>(d_a, d_na);

  checkCudaErrors(cublasSaxpy(cublasHandle, N, d_na, d_Ax, 1, d_r, 1));

  checkCudaErrors(cudaMemcpyAsync(d_r0, d_r1, sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream1));

  checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, d_r1));

  checkCudaErrors(cudaMemcpyAsync(&r1, d_r1, sizeof(float),
                                  cudaMemcpyDeviceToHost, stream1));
  checkCudaErrors(cudaStreamSynchronize(stream1));
  printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
  // First Iteration when k=1 ends
  k++;

#if WITH_GRAPH
  cudaGraph_t initGraph;
  checkCudaErrors(cudaStreamCreate(&streamForGraph));
  checkCudaErrors(cublasSetStream(cublasHandle, stream1));
  checkCudaErrors(cusparseSetStream(cusparseHandle, stream1));
  checkCudaErrors(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

  r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_r0, d_b);
  cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
  checkCudaErrors(cublasSscal(cublasHandle, N, d_b, d_p, 1));
  cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST);
  checkCudaErrors(cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1));
  cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);

  checkCudaErrors(
      cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST));
  checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecp, &beta, vecAx, CUDA_R_32F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

  checkCudaErrors(cudaMemsetAsync(d_dot, 0, sizeof(float), stream1));
  checkCudaErrors(cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, d_dot));

  r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_dot, d_a);

  checkCudaErrors(cublasSaxpy(cublasHandle, N, d_a, d_p, 1, d_x, 1));

  a_minus<<<1, 1, 0, stream1>>>(d_a, d_na);

  checkCudaErrors(cublasSaxpy(cublasHandle, N, d_na, d_Ax, 1, d_r, 1));

  checkCudaErrors(cudaMemcpyAsync(d_r0, d_r1, sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream1));
  checkCudaErrors(cudaMemsetAsync(d_r1, 0, sizeof(float), stream1));

  checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, d_r1));

  checkCudaErrors(cudaMemcpyAsync((float *)&r1, d_r1, sizeof(float),
                                  cudaMemcpyDeviceToHost, stream1));

  checkCudaErrors(cudaStreamEndCapture(stream1, &initGraph));
  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, initGraph, NULL, NULL, 0));
#endif

  checkCudaErrors(cublasSetStream(cublasHandle, stream1));
  checkCudaErrors(cusparseSetStream(cusparseHandle, stream1));

  while (r1 > tol * tol && k <= max_iter) {
#if WITH_GRAPH
    checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
    checkCudaErrors(cudaStreamSynchronize(streamForGraph));
#else
    r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_r0, d_b);
    cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
    checkCudaErrors(cublasSscal(cublasHandle, N, d_b, d_p, 1));

    cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST);
    checkCudaErrors(cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1));

    checkCudaErrors(cusparseSpMV(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
        &beta, vecAx, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
    checkCudaErrors(cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, d_dot));

    r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_dot, d_a);

    checkCudaErrors(cublasSaxpy(cublasHandle, N, d_a, d_p, 1, d_x, 1));

    a_minus<<<1, 1, 0, stream1>>>(d_a, d_na);
    checkCudaErrors(cublasSaxpy(cublasHandle, N, d_na, d_Ax, 1, d_r, 1));

    checkCudaErrors(cudaMemcpyAsync(d_r0, d_r1, sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream1));

    checkCudaErrors(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, d_r1));
    checkCudaErrors(cudaMemcpyAsync((float *)&r1, d_r1, sizeof(float),
                                    cudaMemcpyDeviceToHost, stream1));
    checkCudaErrors(cudaStreamSynchronize(stream1));
#endif
    printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    k++;
  }

#if WITH_GRAPH
  checkCudaErrors(cudaMemcpyAsync(x, d_x, N * sizeof(float),
                                  cudaMemcpyDeviceToHost, streamForGraph));
  checkCudaErrors(cudaStreamSynchronize(streamForGraph));
#else
  checkCudaErrors(cudaMemcpyAsync(x, d_x, N * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream1));
  checkCudaErrors(cudaStreamSynchronize(stream1));
#endif

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

#if WITH_GRAPH
  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaGraphDestroy(initGraph));
  checkCudaErrors(cudaStreamDestroy(streamForGraph));
#endif
  checkCudaErrors(cudaStreamDestroy(stream1));
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

  checkCudaErrors(cudaFreeHost(I));
  checkCudaErrors(cudaFreeHost(J));
  checkCudaErrors(cudaFreeHost(val));
  checkCudaErrors(cudaFreeHost(x));
  free(rhs);
  checkCudaErrors(cudaFree(d_col));
  checkCudaErrors(cudaFree(d_row));
  checkCudaErrors(cudaFree(d_val));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_r));
  checkCudaErrors(cudaFree(d_p));
  checkCudaErrors(cudaFree(d_Ax));

  printf("Test Summary:  Error amount = %f\n", err);
  exit((k <= max_iter) ? 0 : 1);
}
