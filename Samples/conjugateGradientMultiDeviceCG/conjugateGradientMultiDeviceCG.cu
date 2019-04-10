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
 * This sample implements a conjugate gradient solver on multiple GPU using
 * Multi Device Cooperative Groups, also uses Unified Memory optimized using
 * prefetching and usage hints.
 *
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <set>

#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

const char *sSDKname = "conjugateGradientMultiDeviceCG";

#define ENABLE_CPU_DEBUG_CODE 0
#define THREADS_PER_BLOCK 512

__device__ double grid_dot_result = 0.0;

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
                        cg::thread_block &cta,
                        const cg::multi_grid_group &multi_grid) {
  for (int i = multi_grid.thread_rank(); i < num_rows; i += multi_grid.size()) {
    int row_elem = I[i];
    int next_row_elem = I[i + 1];
    int num_elems_this_row = next_row_elem - row_elem;

    float output = 0.0;
    for (int j = 0; j < num_elems_this_row; j++) {
      output += alpha * val[row_elem + j] * inputVecX[J[row_elem + j]];
    }

    outputVecY[i] = output;
  }
}

__device__ void gpuSaxpy(float *x, float *y, float a, int size,
                         const cg::multi_grid_group &multi_grid) {
  for (int i = multi_grid.thread_rank(); i < size; i += multi_grid.size()) {
    y[i] = a * x[i] + y[i];
  }
}

__device__ void gpuDotProduct(float *vecA, float *vecB, int size,
                              const cg::thread_block &cta,
                              const cg::multi_grid_group &multi_grid) {
  __shared__ double tmp[THREADS_PER_BLOCK];

  double temp_sum = 0.0;
  for (int i = multi_grid.thread_rank(); i < size; i += multi_grid.size()) {
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
    atomicAdd(&grid_dot_result, beta);
  }
}

__device__ void gpuCopyVector(float *srcA, float *destB, int size,
                              const cg::multi_grid_group &multi_grid) {
  for (int i = multi_grid.thread_rank(); i < size; i += multi_grid.size()) {
    destB[i] = srcA[i];
  }
}

__device__ void gpuScaleVector(float *vec, float alpha, int size,
                               const cg::multi_grid_group &multi_grid) {
  for (int i = multi_grid.thread_rank(); i < size; i += multi_grid.size()) {
    vec[i] = alpha * vec[i];
  }
}

__device__ void setDotResultToZero(double *dot_result) {
  unsigned long long int *address_as_ull = (unsigned long long int *)dot_result;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS_system(address_as_ull, assumed, 0);

  } while (assumed != old);
}

extern "C" __global__ void multiGpuConjugateGradient(
    int *I, int *J, float *val, float *x, float *Ax, float *p, float *r,
    double *dot_result, int nnz, int N, float tol) {
  cg::thread_block cta = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();
  cg::multi_grid_group multi_grid = cg::this_multi_grid();

  const int max_iter = 10000;

  float alpha = 1.0;
  float alpham1 = -1.0;
  float r0 = 0.0, r1, b, a, na;

  for (int i = multi_grid.thread_rank(); i < N; i += multi_grid.size()) {
    r[i] = 1.0;
    x[i] = 0.0;
  }

  cg::sync(grid);

  gpuSpMV(I, J, val, nnz, N, alpha, x, Ax, cta, multi_grid);

  cg::sync(grid);

  gpuSaxpy(Ax, r, alpham1, N, multi_grid);

  cg::sync(grid);

  gpuDotProduct(r, r, N, cta, multi_grid);

  cg::sync(grid);

  if (grid.thread_rank() == 0) {
    atomicAdd_system(dot_result, grid_dot_result);
    grid_dot_result = 0.0;
  }
  cg::sync(multi_grid);

  r1 = *dot_result;

  int k = 1;
  while (r1 > tol * tol && k <= max_iter) {
    if (k > 1) {
      b = r1 / r0;

      gpuScaleVector(p, b, N, multi_grid);
      cg::sync(grid);
      gpuSaxpy(r, p, alpha, N, multi_grid);
    } else {
      gpuCopyVector(r, p, N, multi_grid);
    }

    cg::sync(multi_grid);

    gpuSpMV(I, J, val, nnz, N, alpha, p, Ax, cta, multi_grid);

    if (multi_grid.thread_rank() == 0) {
      setDotResultToZero(dot_result);
    }
    cg::sync(multi_grid);

    gpuDotProduct(p, Ax, N, cta, multi_grid);

    cg::sync(grid);

    if (grid.thread_rank() == 0) {
      atomicAdd_system(dot_result, grid_dot_result);
      grid_dot_result = 0.0;
    }
    cg::sync(multi_grid);

    a = r1 / *dot_result;

    gpuSaxpy(p, x, a, N, multi_grid);

    na = -a;

    gpuSaxpy(Ax, r, na, N, multi_grid);

    r0 = r1;

    cg::sync(multi_grid);
    if (multi_grid.thread_rank() == 0) {
      setDotResultToZero(dot_result);
    }

    cg::sync(multi_grid);

    gpuDotProduct(r, r, N, cta, multi_grid);

    cg::sync(grid);

    if (grid.thread_rank() == 0) {
      atomicAdd_system(dot_result, grid_dot_result);
      grid_dot_result = 0.0;
    }
    cg::sync(multi_grid);

    r1 = *dot_result;
    k++;
  }
}

void getIdenticalGPUs(int num_of_gpus, std::set<int> &identicalGPUs) {
  int *major_minor =
      reinterpret_cast<int *>(malloc(sizeof(int) * num_of_gpus * 2));
  int foundIdenticalGPUs = 0;

  for (int i = 0; i < num_of_gpus; i++) {
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, i));
    major_minor[i * 2] = deviceProp.major;
    major_minor[i * 2 + 1] = deviceProp.minor;
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", i,
           deviceProp.name, deviceProp.major, deviceProp.minor);
  }

  int maxMajorMinor[2] = {0, 0};

  for (int i = 0; i < num_of_gpus; i++) {
    for (int j = i + 1; j < num_of_gpus; j++) {
      if ((major_minor[i * 2] == major_minor[j * 2]) &&
          (major_minor[i * 2 + 1] == major_minor[j * 2 + 1])) {
        identicalGPUs.insert(i);
        identicalGPUs.insert(j);
        foundIdenticalGPUs = 1;
        if (maxMajorMinor[0] < major_minor[i * 2] &&
            maxMajorMinor[1] < major_minor[i * 2 + 1]) {
          maxMajorMinor[0] = major_minor[i * 2];
          maxMajorMinor[1] = major_minor[i * 2 + 1];
        }
      }
    }
  }

  free(major_minor);
  if (!foundIdenticalGPUs) {
    printf(
        "No Two or more GPUs with same architecture found\nWaiving the "
        "sample\n");
    exit(EXIT_WAIVED);
  }

  std::set<int>::iterator it = identicalGPUs.begin();

  // Iterate over all the identical GPUs found
  while (it != identicalGPUs.end()) {
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, *it));
    // Remove all the GPUs which are less than the best arch available
    if (deviceProp.major != maxMajorMinor[0] &&
        deviceProp.minor != maxMajorMinor[1]) {
      identicalGPUs.erase(it);
    }
    if (!deviceProp.cooperativeMultiDeviceLaunch ||
        !deviceProp.concurrentManagedAccess) {
      identicalGPUs.erase(it);
    }
    it++;
  }

  return;
}

int main(int argc, char **argv) {
  int N = 0, nz = 0, *I = NULL, *J = NULL;
  float *val = NULL;
  const float tol = 1e-5f;
  float *x;
  float rhs = 1.0;
  float r1;
  float *r, *p, *Ax;

  printf("Starting [%s]...\n", sSDKname);

  int num_of_gpus = 0;
  checkCudaErrors(cudaGetDeviceCount(&num_of_gpus));

  if (num_of_gpus <= 1) {
    printf("No. of GPU on node %d\n", num_of_gpus);
    printf("Minimum Two or more GPUs are required to run this sample code\n");
    exit(EXIT_WAIVED);
  }

  std::set<int> identicalGPUs;
  getIdenticalGPUs(num_of_gpus, identicalGPUs);

  if (identicalGPUs.size() <= 1) {
    printf(
        "No Two or more GPUs with same architecture capable of "
        "cooperativeMultiDeviceLaunch & concurrentManagedAccess found. \nWaiving the sample\n");
    exit(EXIT_WAIVED);
  }

  std::set<int>::iterator deviceId = identicalGPUs.begin();

  // We use only 2 GPUs as for input size of N = 10485760*2 two GPUs are enough.
  while (identicalGPUs.size() > 2) {
    identicalGPUs.erase(deviceId);
    deviceId++;
  }
  /* Generate a random tridiagonal symmetric matrix in CSR format */
  N = 10485760 * 2;
  nz = (N - 2) * 3 + 4;

  checkCudaErrors(
      cudaMallocManaged(reinterpret_cast<void **>(&I), sizeof(int) * (N + 1)));
  checkCudaErrors(
      cudaMallocManaged(reinterpret_cast<void **>(&J), sizeof(int) * nz));
  checkCudaErrors(
      cudaMallocManaged(reinterpret_cast<void **>(&val), sizeof(float) * nz));

  float *val_cpu = reinterpret_cast<float *>(malloc(sizeof(float) * nz));

  genTridiag(I, J, val_cpu, N, nz);

  memcpy(val, val_cpu, sizeof(float) * nz);
  checkCudaErrors(
      cudaMemAdvise(I, sizeof(int) * (N + 1), cudaMemAdviseSetReadMostly, 0));
  checkCudaErrors(
      cudaMemAdvise(J, sizeof(int) * nz, cudaMemAdviseSetReadMostly, 0));
  checkCudaErrors(
      cudaMemAdvise(val, sizeof(float) * nz, cudaMemAdviseSetReadMostly, 0));

  checkCudaErrors(
      cudaMallocManaged(reinterpret_cast<void **>(&x), sizeof(float) * N));

  double *dot_result;
  checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&dot_result),
                                    sizeof(double)));

  checkCudaErrors(cudaMemset(dot_result, 0.0, sizeof(double)));

  // temp memory for ConjugateGradient
  checkCudaErrors(
      cudaMallocManaged(reinterpret_cast<void **>(&r), N * sizeof(float)));
  checkCudaErrors(
      cudaMallocManaged(reinterpret_cast<void **>(&p), N * sizeof(float)));
  checkCudaErrors(
      cudaMallocManaged(reinterpret_cast<void **>(&Ax), N * sizeof(float)));

  std::cout << "\nRunning on GPUs = " << identicalGPUs.size() << std::endl;
  cudaStream_t *nStreams = reinterpret_cast<cudaStream_t *>(
      malloc(sizeof(cudaStream_t) * identicalGPUs.size()));

  void *kernelArgs[] = {
      (void *)&I,  (void *)&J, (void *)&val, (void *)&x,
      (void *)&Ax, (void *)&p, (void *)&r,   (void *)&dot_result,
      (void *)&nz, (void *)&N, (void *)&tol,
  };

  int sMemSize = sizeof(double) * THREADS_PER_BLOCK;
  int numBlocksPerSm = 0;
  int numThreads = THREADS_PER_BLOCK;

  deviceId = identicalGPUs.begin();
  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaSetDevice(*deviceId));
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, *deviceId));

  checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, multiGpuConjugateGradient, numThreads, sMemSize));

  int numSms = deviceProp.multiProcessorCount;
  dim3 dimGrid(numSms * numBlocksPerSm, 1, 1),
      dimBlock(THREADS_PER_BLOCK, 1, 1);

  int device_count = 0;

  int totalThreadsPerGPU = numSms * numBlocksPerSm * THREADS_PER_BLOCK;

  while (deviceId != identicalGPUs.end()) {
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaSetDevice(*deviceId));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, *deviceId));
    checkCudaErrors(cudaStreamCreate(&nStreams[device_count]));

    if (deviceProp.concurrentManagedAccess) {
      int perGPUIter = N / (totalThreadsPerGPU * identicalGPUs.size());
      int offset_Ax = device_count * totalThreadsPerGPU;
      int offset_r = device_count * totalThreadsPerGPU;
      int offset_p = device_count * totalThreadsPerGPU;
      int offset_x = device_count * totalThreadsPerGPU;

      checkCudaErrors(cudaMemPrefetchAsync(I, sizeof(int) * N, *deviceId,
                                           nStreams[device_count]));
      checkCudaErrors(cudaMemPrefetchAsync(val, sizeof(float) * nz, *deviceId,
                                           nStreams[device_count]));
      checkCudaErrors(cudaMemPrefetchAsync(J, sizeof(float) * nz, *deviceId,
                                           nStreams[device_count]));

      if (offset_Ax <= N) {
        for (int i = 0; i < perGPUIter; i++) {
          cudaMemAdvise(Ax + offset_Ax, sizeof(float) * totalThreadsPerGPU,
                        cudaMemAdviseSetPreferredLocation, *deviceId);
          cudaMemAdvise(r + offset_r, sizeof(float) * totalThreadsPerGPU,
                        cudaMemAdviseSetPreferredLocation, *deviceId);
          cudaMemAdvise(x + offset_x, sizeof(float) * totalThreadsPerGPU,
                        cudaMemAdviseSetPreferredLocation, *deviceId);
          cudaMemAdvise(p + offset_p, sizeof(float) * totalThreadsPerGPU,
                        cudaMemAdviseSetPreferredLocation, *deviceId);

          cudaMemAdvise(Ax + offset_Ax, sizeof(float) * totalThreadsPerGPU,
                        cudaMemAdviseSetAccessedBy, *deviceId);
          cudaMemAdvise(r + offset_r, sizeof(float) * totalThreadsPerGPU,
                        cudaMemAdviseSetAccessedBy, *deviceId);
          cudaMemAdvise(p + offset_p, sizeof(float) * totalThreadsPerGPU,
                        cudaMemAdviseSetAccessedBy, *deviceId);
          cudaMemAdvise(x + offset_x, sizeof(float) * totalThreadsPerGPU,
                        cudaMemAdviseSetAccessedBy, *deviceId);

          offset_Ax += totalThreadsPerGPU * identicalGPUs.size();
          offset_r += totalThreadsPerGPU * identicalGPUs.size();
          offset_p += totalThreadsPerGPU * identicalGPUs.size();
          offset_x += totalThreadsPerGPU * identicalGPUs.size();

          if (offset_Ax >= N) {
            break;
          }
        }
      }
    }
    device_count++;
    deviceId++;
  }

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

  printf("Total threads per GPU = %d numBlocksPerSm  = %d\n",
         numSms * numBlocksPerSm * THREADS_PER_BLOCK, numBlocksPerSm);
  cudaLaunchParams *launchParamsList = reinterpret_cast<cudaLaunchParams *>(
      malloc(sizeof(cudaLaunchParams) * identicalGPUs.size()));
  for (int i = 0; i < identicalGPUs.size(); i++) {
    launchParamsList[i].func =
        reinterpret_cast<void *>(multiGpuConjugateGradient);
    launchParamsList[i].gridDim = dimGrid;
    launchParamsList[i].blockDim = dimBlock;
    launchParamsList[i].sharedMem = sMemSize;
    launchParamsList[i].stream = nStreams[i];
    launchParamsList[i].args = kernelArgs;
  }

  printf("Launching kernel\n");
  checkCudaErrors(cudaLaunchCooperativeKernelMultiDevice(
      launchParamsList, identicalGPUs.size(),
      cudaCooperativeLaunchMultiDeviceNoPreSync |
          cudaCooperativeLaunchMultiDeviceNoPostSync));

  if (deviceProp.concurrentManagedAccess) {
    checkCudaErrors(
        cudaMemPrefetchAsync(x, sizeof(float) * N, cudaCpuDeviceId));
    checkCudaErrors(
        cudaMemPrefetchAsync(dot_result, sizeof(double), cudaCpuDeviceId));
  }

  deviceId = identicalGPUs.begin();
  device_count = 0;
  while (deviceId != identicalGPUs.end()) {
    checkCudaErrors(cudaSetDevice(*deviceId));
    checkCudaErrors(cudaStreamSynchronize(nStreams[device_count++]));
    deviceId++;
  }

  r1 = *dot_result;

  printf("GPU Final, residual = %e \n  ", sqrt(r1));

#if ENABLE_CPU_DEBUG_CODE
  cpuConjugateGrad(I, J, val, x_cpu, Ax_cpu, p_cpu, r_cpu, nz, N, tol);
#endif

  float rsum, diff, err = 0.0;

  for (int i = 0; i < N; i++) {
    rsum = 0.0;

    for (int j = I[i]; j < I[i + 1]; j++) {
      rsum += val_cpu[j] * x[J[j]];
    }

    diff = fabs(rsum - rhs);

    if (diff > err) {
      err = diff;
    }
  }

  checkCudaErrors(cudaFree(I));
  checkCudaErrors(cudaFree(J));
  checkCudaErrors(cudaFree(val));
  checkCudaErrors(cudaFree(x));
  checkCudaErrors(cudaFree(r));
  checkCudaErrors(cudaFree(p));
  checkCudaErrors(cudaFree(Ax));
  checkCudaErrors(cudaFree(dot_result));
  free(val_cpu);

#if ENABLE_CPU_DEBUG_CODE
  free(Ax_cpu);
  free(r_cpu);
  free(p_cpu);
  free(x_cpu);
#endif

  printf("Test Summary:  Error amount = %f \n", err);
  fprintf(stdout, "&&&& conjugateGradientMultiDeviceCG %s\n",
          (sqrt(r1) < tol) ? "PASSED" : "FAILED");
  exit((sqrt(r1) < tol) ? EXIT_SUCCESS : EXIT_FAILURE);
}
