/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
 * This sample implements a conjugate gradient solver on GPU using
 * Multi Block Cooperative Groups, also uses Unified Memory.
 *
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

const char *sSDKname = "conjugateGradientMultiBlockCG";

#define ENABLE_CPU_DEBUG_CODE 0
#define THREADS_PER_BLOCK 512

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz) {
  I[0] = 0, J[0] = 0, J[1] = 1;
  val[0] = static_cast<float>(rand()) / RAND_MAX + 10.0f;
  val[1] = static_cast<float>(rand()) / RAND_MAX;
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
    val[start + 1] = static_cast<float>(rand()) / RAND_MAX + 10.0f;

    if (i < N - 1) {
      val[start + 2] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  I[N] = nz;
}

// I - contains location of the given non-zero element in the row of the matrix
// J - contains location of the given non-zero element in the column of the
// matrix val - contains values of the given non-zero elements of the matrix
// inputVecX - input vector to be multiplied
// outputVecY - resultant vector
void cpuSpMV(int *I, int *J, float *val, int nnz, int num_rows, float alpha,
             float *inputVecX, float *outputVecY) {
  for (int i = 0; i < num_rows; i++) {
    int num_elems_this_row = I[i + 1] - I[i];

    float output = 0.0;
    for (int j = 0; j < num_elems_this_row; j++) {
      output += alpha * val[I[i] + j] * inputVecX[J[I[i] + j]];
    }
    outputVecY[i] = output;
  }

  return;
}

double dotProduct(float *vecA, float *vecB, int size) {
  double result = 0.0;

  for (int i = 0; i < size; i++) {
    result = result + (vecA[i] * vecB[i]);
  }

  return result;
}

void scaleVector(float *vec, float alpha, int size) {
  for (int i = 0; i < size; i++) {
    vec[i] = alpha * vec[i];
  }
}

void saxpy(float *x, float *y, float a, int size) {
  for (int i = 0; i < size; i++) {
    y[i] = a * x[i] + y[i];
  }
}

void cpuConjugateGrad(int *I, int *J, float *val, float *x, float *Ax, float *p,
                      float *r, int nnz, int N, float tol) {
  int max_iter = 10000;

  float alpha = 1.0;
  float alpham1 = -1.0;
  float r0 = 0.0, b, a, na;

  cpuSpMV(I, J, val, nnz, N, alpha, x, Ax);
  saxpy(Ax, r, alpham1, N);

  float r1 = dotProduct(r, r, N);

  int k = 1;

  while (r1 > tol * tol && k <= max_iter) {
    if (k > 1) {
      b = r1 / r0;
      scaleVector(p, b, N);

      saxpy(r, p, alpha, N);
    } else {
      for (int i = 0; i < N; i++) p[i] = r[i];
    }

    cpuSpMV(I, J, val, nnz, N, alpha, p, Ax);

    float dot = dotProduct(p, Ax, N);
    a = r1 / dot;

    saxpy(p, x, a, N);
    na = -a;
    saxpy(Ax, r, na, N);

    r0 = r1;
    r1 = dotProduct(r, r, N);

    printf("\nCPU code iteration = %3d, residual = %e\n", k, sqrt(r1));
    k++;
  }
}

__device__ void gpuSpMV(int *I, int *J, float *val, int nnz, int num_rows,
                        float alpha, float *inputVecX, float *outputVecY,
                        cg::thread_block &cta, const cg::grid_group &grid) {
  for (int i = grid.thread_rank(); i < num_rows; i += grid.size()) {
    int row_elem = I[i];
    int next_row_elem = I[i + 1];
    int num_elems_this_row = next_row_elem - row_elem;

    float output = 0.0;
    for (int j = 0; j < num_elems_this_row; j++) {
      // I or J or val arrays - can be put in shared memory
      // as the access is random and reused in next calls of gpuSpMV function.
      output += alpha * val[row_elem + j] * inputVecX[J[row_elem + j]];
    }

    outputVecY[i] = output;
  }
}

__device__ void gpuSaxpy(float *x, float *y, float a, int size,
                         const cg::grid_group &grid) {
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    y[i] = a * x[i] + y[i];
  }
}

__device__ void gpuDotProduct(float *vecA, float *vecB, double *result,
                              int size, const cg::thread_block &cta,
                              const cg::grid_group &grid) {
  __shared__ double tmp[THREADS_PER_BLOCK];

  double temp_sum = 0.0;
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    temp_sum += static_cast<double>(vecA[i] * vecB[i]);
  }
  tmp[cta.thread_rank()] = temp_sum;

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  double beta = temp_sum;
  double temp;

  for (int i = tile32.size() / 2; i > 0; i >>= 1) {
    if (tile32.thread_rank() < i) {
      temp = tmp[cta.thread_rank() + i];
      beta += temp;
      tmp[cta.thread_rank()] = beta;
    }
    cg::sync(tile32);
  }
  cg::sync(cta);

  if (cta.thread_rank() == 0) {
    beta = 0.0;
    for (int i = 0; i < cta.size(); i += tile32.size()) {
      beta += tmp[i];
    }
    atomicAdd(result, beta);
  }
}

__device__ void gpuCopyVector(float *srcA, float *destB, int size,
                              const cg::grid_group &grid) {
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    destB[i] = srcA[i];
  }
}

__device__ void gpuScaleVector(float *vec, float alpha, int size,
                               const cg::grid_group &grid) {
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    vec[i] = alpha * vec[i];
  }
}

extern "C" __global__ void gpuConjugateGradient(int *I, int *J, float *val,
                                                float *x, float *Ax, float *p,
                                                float *r, double *dot_result,
                                                int nnz, int N, float tol) {
  cg::thread_block cta = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();

  int max_iter = 10000;

  float alpha = 1.0;
  float alpham1 = -1.0;
  float r0 = 0.0, r1, b, a, na;

  gpuSpMV(I, J, val, nnz, N, alpha, x, Ax, cta, grid);

  cg::sync(grid);

  gpuSaxpy(Ax, r, alpham1, N, grid);

  cg::sync(grid);

  gpuDotProduct(r, r, dot_result, N, cta, grid);

  cg::sync(grid);

  r1 = *dot_result;

  int k = 1;
  while (r1 > tol * tol && k <= max_iter) {
    if (k > 1) {
      b = r1 / r0;
      gpuScaleVector(p, b, N, grid);

      cg::sync(grid);
      gpuSaxpy(r, p, alpha, N, grid);
    } else {
      gpuCopyVector(r, p, N, grid);
    }

    cg::sync(grid);

    gpuSpMV(I, J, val, nnz, N, alpha, p, Ax, cta, grid);

    if (threadIdx.x == 0 && blockIdx.x == 0) *dot_result = 0.0;

    cg::sync(grid);

    gpuDotProduct(p, Ax, dot_result, N, cta, grid);

    cg::sync(grid);

    a = r1 / *dot_result;

    gpuSaxpy(p, x, a, N, grid);
    na = -a;
    gpuSaxpy(Ax, r, na, N, grid);

    r0 = r1;

    cg::sync(grid);
    if (threadIdx.x == 0 && blockIdx.x == 0) *dot_result = 0.0;

    cg::sync(grid);

    gpuDotProduct(r, r, dot_result, N, cta, grid);

    cg::sync(grid);

    r1 = *dot_result;
    k++;
  }
}

bool areAlmostEqual(float a, float b, float maxRelDiff) {
  float diff = fabsf(a - b);
  float abs_a = fabsf(a);
  float abs_b = fabsf(b);
  float largest = abs_a > abs_b ? abs_a : abs_b;

  if (diff <= largest * maxRelDiff) {
    return true;
  } else {
    printf("maxRelDiff = %.8e\n", maxRelDiff);
    printf(
        "diff %.8e > largest * maxRelDiff %.8e therefore %.8e and %.8e are not "
        "same\n",
        diff, largest * maxRelDiff, a, b);
    return false;
  }
}

int main(int argc, char **argv) {
  int N = 0, nz = 0, *I = NULL, *J = NULL;
  float *val = NULL;
  const float tol = 1e-5f;
  float *x;
  float *rhs;
  float r1;
  float *r, *p, *Ax;
  cudaEvent_t start, stop;

  printf("Starting [%s]...\n", sSDKname);

  // This will pick the best possible CUDA capable device
  cudaDeviceProp deviceProp;
  int devID = findCudaDevice(argc, (const char **)argv);
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

  if (!deviceProp.managedMemory) {
    // This sample requires being run on a device that supports Unified Memory
    fprintf(stderr, "Unified Memory not supported on this device\n");
    exit(EXIT_WAIVED);
  }

  // This sample requires being run on a device that supports Cooperative Kernel
  // Launch
  if (!deviceProp.cooperativeLaunch) {
    printf(
        "\nSelected GPU (%d) does not support Cooperative Kernel Launch, "
        "Waiving the run\n",
        devID);
    exit(EXIT_WAIVED);
  }

  // Statistics about the GPU device
  printf(
      "> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
      deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

  /* Generate a random tridiagonal symmetric matrix in CSR format */
  N = 1048576;
  nz = (N - 2) * 3 + 4;

  cudaMallocManaged(reinterpret_cast<void **>(&I), sizeof(int) * (N + 1));
  cudaMallocManaged(reinterpret_cast<void **>(&J), sizeof(int) * nz);
  cudaMallocManaged(reinterpret_cast<void **>(&val), sizeof(float) * nz);

  genTridiag(I, J, val, N, nz);

  cudaMallocManaged(reinterpret_cast<void **>(&x), sizeof(float) * N);
  cudaMallocManaged(reinterpret_cast<void **>(&rhs), sizeof(float) * N);

  double *dot_result;

  cudaMallocManaged(reinterpret_cast<void **>(&dot_result), sizeof(double));

  *dot_result = 0.0;

  // temp memory for CG
  checkCudaErrors(
      cudaMallocManaged(reinterpret_cast<void **>(&r), N * sizeof(float)));
  checkCudaErrors(
      cudaMallocManaged(reinterpret_cast<void **>(&p), N * sizeof(float)));
  checkCudaErrors(
      cudaMallocManaged(reinterpret_cast<void **>(&Ax), N * sizeof(float)));

  cudaDeviceSynchronize();

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

#if ENABLE_CPU_DEBUG_CODE
  float *Ax_cpu = reinterpret_cast<float *>(malloc(sizeof(float) * N));
  float *r_cpu = reinterpret_cast<float *>(malloc(sizeof(float) * N));
  float *p_cpu = reinterpret_cast<float *>(malloc(sizeof(float) * N));
  float *x_cpu = reinterpret_cast<float *>(malloc(sizeof(float) * N));

  for (int i = 0; i < N; i++) {
    r_cpu[i] = 1.0;
    Ax_cpu[i] = x_cpu[i] = 0.0;
  }

#endif

  for (int i = 0; i < N; i++) {
    r[i] = rhs[i] = 1.0;
    x[i] = 0.0;
  }

  void *kernelArgs[] = {
      (void *)&I,  (void *)&J, (void *)&val, (void *)&x,
      (void *)&Ax, (void *)&p, (void *)&r,   (void *)&dot_result,
      (void *)&nz, (void *)&N, (void *)&tol,
  };

  int sMemSize = sizeof(double) * THREADS_PER_BLOCK;
  int numBlocksPerSm = 0;
  int numThreads = THREADS_PER_BLOCK;

  checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, gpuConjugateGradient, numThreads, sMemSize));

  int numSms = deviceProp.multiProcessorCount;
  dim3 dimGrid(numSms * numBlocksPerSm, 1, 1),
      dimBlock(THREADS_PER_BLOCK, 1, 1);
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaLaunchCooperativeKernel((void *)gpuConjugateGradient,
                                              dimGrid, dimBlock, kernelArgs,
                                              sMemSize, NULL));
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaDeviceSynchronize());

  float time;
  checkCudaErrors(cudaEventElapsedTime(&time, start, stop));

  r1 = *dot_result;

  printf("GPU Final, residual = %e, kernel execution time = %f ms\n", sqrt(r1),
         time);

#if ENABLE_CPU_DEBUG_CODE
  cpuConjugateGrad(I, J, val, x_cpu, Ax_cpu, p_cpu, r_cpu, nz, N, tol);
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

  checkCudaErrors(cudaFree(I));
  checkCudaErrors(cudaFree(J));
  checkCudaErrors(cudaFree(val));
  checkCudaErrors(cudaFree(x));
  checkCudaErrors(cudaFree(rhs));
  checkCudaErrors(cudaFree(r));
  checkCudaErrors(cudaFree(p));
  checkCudaErrors(cudaFree(Ax));
  checkCudaErrors(cudaFree(dot_result));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

#if ENABLE_CPU_DEBUG_CODE
  free(Ax_cpu);
  free(r_cpu);
  free(p_cpu);
  free(x_cpu);
#endif

  printf("Test Summary:  Error amount = %f \n", err);
  fprintf(stdout, "&&&& conjugateGradientMultiBlockCG %s\n",
          (sqrt(r1) < tol) ? "PASSED" : "FAILED");
  exit((sqrt(r1) < tol) ? EXIT_SUCCESS : EXIT_FAILURE);
}
