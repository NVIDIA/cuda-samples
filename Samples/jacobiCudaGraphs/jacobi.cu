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

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <vector>
#include "jacobi.h"

namespace cg = cooperative_groups;

// 8 Rows of square-matrix A processed by each CTA.
// This can be max 32 and only power of 2 (i.e., 2/4/8/16/32).
#define ROWS_PER_CTA 8

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

static __global__ void JacobiMethod(const float *A, const double *b,
                                    const float conv_threshold, double *x,
                                    double *x_new, double *sum) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ double x_shared[N_ROWS];  // N_ROWS == n
  __shared__ double b_shared[ROWS_PER_CTA + 1];

  for (int i = threadIdx.x; i < N_ROWS; i += blockDim.x) {
    x_shared[i] = x[i];
  }

  if (threadIdx.x < ROWS_PER_CTA) {
    int k = threadIdx.x;
    for (int i = k + (blockIdx.x * ROWS_PER_CTA);
         (k < ROWS_PER_CTA) && (i < N_ROWS);
         k += ROWS_PER_CTA, i += ROWS_PER_CTA) {
      b_shared[i % (ROWS_PER_CTA + 1)] = b[i];
    }
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  for (int k = 0, i = blockIdx.x * ROWS_PER_CTA;
       (k < ROWS_PER_CTA) && (i < N_ROWS); k++, i++) {
    double rowThreadSum = 0.0;
    for (int j = threadIdx.x; j < N_ROWS; j += blockDim.x) {
      rowThreadSum += (A[i * N_ROWS + j] * x_shared[j]);
    }

    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      rowThreadSum += tile32.shfl_down(rowThreadSum, offset);
    }

    if (tile32.thread_rank() == 0) {
      atomicAdd(&b_shared[i % (ROWS_PER_CTA + 1)], -rowThreadSum);
    }
  }

  cg::sync(cta);

  if (threadIdx.x < ROWS_PER_CTA) {
    cg::thread_block_tile<ROWS_PER_CTA> tile8 =
        cg::tiled_partition<ROWS_PER_CTA>(cta);
    double temp_sum = 0.0;

    int k = threadIdx.x;

    for (int i = k + (blockIdx.x * ROWS_PER_CTA);
         (k < ROWS_PER_CTA) && (i < N_ROWS);
         k += ROWS_PER_CTA, i += ROWS_PER_CTA) {
      double dx = b_shared[i % (ROWS_PER_CTA + 1)];
      dx /= A[i * N_ROWS + i];

      x_new[i] = (x_shared[i] + dx);
      temp_sum += fabs(dx);
    }

    for (int offset = tile8.size() / 2; offset > 0; offset /= 2) {
      temp_sum += tile8.shfl_down(temp_sum, offset);
    }

    if (tile8.thread_rank() == 0) {
      atomicAdd(sum, temp_sum);
    }
  }
}

// Thread block size for finalError kernel should be multiple of 32
static __global__ void finalError(double *x, double *g_sum) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ double warpSum[];
  double sum = 0.0;

  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = globalThreadId; i < N_ROWS; i += blockDim.x * gridDim.x) {
    double d = x[i] - 1.0;
    sum += fabs(d);
  }

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
    sum += tile32.shfl_down(sum, offset);
  }

  if (tile32.thread_rank() == 0) {
    warpSum[threadIdx.x / warpSize] = sum;
  }

  cg::sync(cta);

  double blockSum = 0.0;
  if (threadIdx.x < (blockDim.x / warpSize)) {
    blockSum = warpSum[threadIdx.x];
  }

  if (threadIdx.x < 32) {
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      blockSum += tile32.shfl_down(blockSum, offset);
    }
    if (tile32.thread_rank() == 0) {
      atomicAdd(g_sum, blockSum);
    }
  }
}

double JacobiMethodGpuCudaGraphExecKernelSetParams(
    const float *A, const double *b, const float conv_threshold,
    const int max_iter, double *x, double *x_new, cudaStream_t stream) {
  // CTA size
  dim3 nthreads(256, 1, 1);
  // grid size
  dim3 nblocks((N_ROWS / ROWS_PER_CTA) + 2, 1, 1);
  cudaGraph_t graph;
  cudaGraphExec_t graphExec = NULL;

  double sum = 0.0;
  double *d_sum = NULL;
  checkCudaErrors(cudaMalloc(&d_sum, sizeof(double)));

  std::vector<cudaGraphNode_t> nodeDependencies;
  cudaGraphNode_t memcpyNode, jacobiKernelNode, memsetNode;
  cudaMemcpy3DParms memcpyParams = {0};
  cudaMemsetParams memsetParams = {0};

  memsetParams.dst = (void *)d_sum;
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  // elementSize can be max 4 bytes, so we take sizeof(float) and width=2
  memsetParams.elementSize = sizeof(float);
  memsetParams.width = 2;
  memsetParams.height = 1;

  checkCudaErrors(cudaGraphCreate(&graph, 0));
  checkCudaErrors(
      cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));
  nodeDependencies.push_back(memsetNode);

  cudaKernelNodeParams NodeParams0, NodeParams1;
  NodeParams0.func = (void *)JacobiMethod;
  NodeParams0.gridDim = nblocks;
  NodeParams0.blockDim = nthreads;
  NodeParams0.sharedMemBytes = 0;
  void *kernelArgs0[6] = {(void *)&A, (void *)&b,     (void *)&conv_threshold,
                          (void *)&x, (void *)&x_new, (void *)&d_sum};
  NodeParams0.kernelParams = kernelArgs0;
  NodeParams0.extra = NULL;

  checkCudaErrors(
      cudaGraphAddKernelNode(&jacobiKernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &NodeParams0));

  nodeDependencies.clear();
  nodeDependencies.push_back(jacobiKernelNode);

  memcpyParams.srcArray = NULL;
  memcpyParams.srcPos = make_cudaPos(0, 0, 0);
  memcpyParams.srcPtr = make_cudaPitchedPtr(d_sum, sizeof(double), 1, 1);
  memcpyParams.dstArray = NULL;
  memcpyParams.dstPos = make_cudaPos(0, 0, 0);
  memcpyParams.dstPtr = make_cudaPitchedPtr(&sum, sizeof(double), 1, 1);
  memcpyParams.extent = make_cudaExtent(sizeof(double), 1, 1);
  memcpyParams.kind = cudaMemcpyDeviceToHost;

  checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &memcpyParams));

  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  NodeParams1.func = (void *)JacobiMethod;
  NodeParams1.gridDim = nblocks;
  NodeParams1.blockDim = nthreads;
  NodeParams1.sharedMemBytes = 0;
  void *kernelArgs1[6] = {(void *)&A,     (void *)&b, (void *)&conv_threshold,
                          (void *)&x_new, (void *)&x, (void *)&d_sum};
  NodeParams1.kernelParams = kernelArgs1;
  NodeParams1.extra = NULL;

  int k = 0;
  for (k = 0; k < max_iter; k++) {
    checkCudaErrors(cudaGraphExecKernelNodeSetParams(
        graphExec, jacobiKernelNode,
        ((k & 1) == 0) ? &NodeParams0 : &NodeParams1));
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    if (sum <= conv_threshold) {
      checkCudaErrors(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
      nblocks.x = (N_ROWS / nthreads.x) + 1;
      size_t sharedMemSize = ((nthreads.x / 32) + 1) * sizeof(double);
      if ((k & 1) == 0) {
        finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x_new, d_sum);
      } else {
        finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x, d_sum);
      }

      checkCudaErrors(cudaMemcpyAsync(&sum, d_sum, sizeof(double),
                                      cudaMemcpyDeviceToHost, stream));
      checkCudaErrors(cudaStreamSynchronize(stream));
      printf("GPU iterations : %d\n", k + 1);
      printf("GPU error : %.3e\n", sum);
      break;
    }
  }

  checkCudaErrors(cudaFree(d_sum));
  return sum;
}

double JacobiMethodGpuCudaGraphExecUpdate(const float *A, const double *b,
                                          const float conv_threshold,
                                          const int max_iter, double *x,
                                          double *x_new, cudaStream_t stream) {
  // CTA size
  dim3 nthreads(256, 1, 1);
  // grid size
  dim3 nblocks((N_ROWS / ROWS_PER_CTA) + 2, 1, 1);
  cudaGraph_t graph;
  cudaGraphExec_t graphExec = NULL;

  double sum = 0.0;
  double *d_sum;
  checkCudaErrors(cudaMalloc(&d_sum, sizeof(double)));

  int k = 0;
  for (k = 0; k < max_iter; k++) {
    checkCudaErrors(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    checkCudaErrors(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
    if ((k & 1) == 0) {
      JacobiMethod<<<nblocks, nthreads, 0, stream>>>(A, b, conv_threshold, x,
                                                     x_new, d_sum);
    } else {
      JacobiMethod<<<nblocks, nthreads, 0, stream>>>(A, b, conv_threshold,
                                                     x_new, x, d_sum);
    }
    checkCudaErrors(cudaMemcpyAsync(&sum, d_sum, sizeof(double),
                                    cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamEndCapture(stream, &graph));

    if (graphExec == NULL) {
      checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    } else {
      cudaGraphExecUpdateResult updateResult_out;
      checkCudaErrors(
          cudaGraphExecUpdate(graphExec, graph, NULL, &updateResult_out));
      if (updateResult_out != cudaGraphExecUpdateSuccess) {
        if (graphExec != NULL) {
          checkCudaErrors(cudaGraphExecDestroy(graphExec));
        }
        printf("k = %d graph update failed with error - %d\n", k,
               updateResult_out);
        checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
      }
    }
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    if (sum <= conv_threshold) {
      checkCudaErrors(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
      nblocks.x = (N_ROWS / nthreads.x) + 1;
      size_t sharedMemSize = ((nthreads.x / 32) + 1) * sizeof(double);
      if ((k & 1) == 0) {
        finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x_new, d_sum);
      } else {
        finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x, d_sum);
      }

      checkCudaErrors(cudaMemcpyAsync(&sum, d_sum, sizeof(double),
                                      cudaMemcpyDeviceToHost, stream));
      checkCudaErrors(cudaStreamSynchronize(stream));
      printf("GPU iterations : %d\n", k + 1);
      printf("GPU error : %.3e\n", sum);
      break;
    }
  }

  checkCudaErrors(cudaFree(d_sum));
  return sum;
}

double JacobiMethodGpu(const float *A, const double *b,
                       const float conv_threshold, const int max_iter,
                       double *x, double *x_new, cudaStream_t stream) {
  // CTA size
  dim3 nthreads(256, 1, 1);
  // grid size
  dim3 nblocks((N_ROWS / ROWS_PER_CTA) + 2, 1, 1);

  double sum = 0.0;
  double *d_sum;
  checkCudaErrors(cudaMalloc(&d_sum, sizeof(double)));
  int k = 0;

  for (k = 0; k < max_iter; k++) {
    checkCudaErrors(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
    if ((k & 1) == 0) {
      JacobiMethod<<<nblocks, nthreads, 0, stream>>>(A, b, conv_threshold, x,
                                                     x_new, d_sum);
    } else {
      JacobiMethod<<<nblocks, nthreads, 0, stream>>>(A, b, conv_threshold,
                                                     x_new, x, d_sum);
    }
    checkCudaErrors(cudaMemcpyAsync(&sum, d_sum, sizeof(double),
                                    cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    if (sum <= conv_threshold) {
      checkCudaErrors(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
      nblocks.x = (N_ROWS / nthreads.x) + 1;
      size_t sharedMemSize = ((nthreads.x / 32) + 1) * sizeof(double);
      if ((k & 1) == 0) {
        finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x_new, d_sum);
      } else {
        finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x, d_sum);
      }

      checkCudaErrors(cudaMemcpyAsync(&sum, d_sum, sizeof(double),
                                      cudaMemcpyDeviceToHost, stream));
      checkCudaErrors(cudaStreamSynchronize(stream));
      printf("GPU iterations : %d\n", k + 1);
      printf("GPU error : %.3e\n", sum);
      break;
    }
  }

  checkCudaErrors(cudaFree(d_sum));
  return sum;
}
