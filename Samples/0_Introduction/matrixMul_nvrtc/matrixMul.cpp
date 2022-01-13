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

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include "nvrtc_helper.h"

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

void constantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int matrixMultiply(int argc, char **argv, int block_size, dim3 &dimsA,
                   dim3 &dimsB) {
  // Allocate host memory for matrices A and B
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A = (float *)malloc(mem_size_A);
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B = (float *)malloc(mem_size_B);

  // Initialize host memory
  const float valB = 0.01f;
  constantInit(h_A, size_A, 1.0f);
  constantInit(h_B, size_B, valB);

  // Allocate device memory
  CUdeviceptr d_A, d_B, d_C;

  char *cubin, *kernel_file;
  size_t cubinSize;

  kernel_file = sdkFindFilePath("matrixMul_kernel.cu", argv[0]);
  compileFileToCUBIN(kernel_file, argc, argv, &cubin, &cubinSize, 1);

  CUmodule module = loadCUBIN(cubin, argc, argv);

  // Allocate host matrix C
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float *h_C = (float *)malloc(mem_size_C);

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cuMemAlloc(&d_A, mem_size_A));
  checkCudaErrors(cuMemAlloc(&d_B, mem_size_B));
  checkCudaErrors(cuMemAlloc(&d_C, mem_size_C));

  // copy host memory to device
  checkCudaErrors(cuMemcpyHtoD(d_A, h_A, mem_size_A));
  checkCudaErrors(cuMemcpyHtoD(d_B, h_B, mem_size_B));

  // Setup execution parameters
  dim3 threads(block_size, block_size);
  dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

  // Create and start timer
  printf("Computing result using CUDA Kernel...\n");

  CUfunction kernel_addr;
  if (block_size == 16) {
    checkCudaErrors(
        cuModuleGetFunction(&kernel_addr, module, "matrixMulCUDA_block16"));
  } else {
    checkCudaErrors(
        cuModuleGetFunction(&kernel_addr, module, "matrixMulCUDA_block32"));
  }

  void *arr[] = {(void *)&d_C, (void *)&d_A, (void *)&d_B, (void *)&dimsA.x,
                 (void *)&dimsB.x};

  // Execute the kernel
  int nIter = 300;

  for (int j = 0; j < nIter; j++) {
    checkCudaErrors(
        cuLaunchKernel(kernel_addr, grid.x, grid.y, grid.z, /* grid dim */
                       threads.x, threads.y, threads.z,     /* block dim */
                       0, 0,    /* shared mem, stream */
                       &arr[0], /* arguments */
                       0));

    checkCudaErrors(cuCtxSynchronize());
  }

  // Copy result from device to host
  checkCudaErrors(cuMemcpyDtoH(h_C, d_C, mem_size_C));

  printf("Checking computed result for correctness: ");

  bool correct = true;

  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps

  double eps = 1.e-6;  // machine zero

  for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++) {
    double abs_err = fabs(h_C[i] - (dimsA.x * valB));
    double dot_length = dimsA.x;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;

    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
             h_C[i], dimsA.x * valB, eps);
      correct = false;
    }
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  printf(
      "\nNOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n");

  // Clean up memory
  free(h_A);
  free(h_B);
  free(h_C);

  checkCudaErrors(cuMemFree(d_A));
  checkCudaErrors(cuMemFree(d_B));
  checkCudaErrors(cuMemFree(d_C));

  if (correct) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}

/**
 * Program main
 */

int main(int argc, char **argv) {
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
    printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
    printf(
        "  Note: Outer matrix dimensions of A & B matrices must be equal.\n");

    exit(EXIT_SUCCESS);
  }

  int block_size = 32;

  // original:
  dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
  dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

  // reduce sizes to avoid running out of memory
  // dim3 dimsA(32,32, 1);
  // dim3 dimsB(32,32,1);

  // width of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
    dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
  }

  // height of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
    dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
  }

  // width of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
    dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
  }

  // height of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
    dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
  }

  if (dimsA.x != dimsB.y) {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }

  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x,
         dimsB.y);

  int matrix_result = matrixMultiply(argc, argv, block_size, dimsA, dimsB);

  exit(matrix_result);
}
