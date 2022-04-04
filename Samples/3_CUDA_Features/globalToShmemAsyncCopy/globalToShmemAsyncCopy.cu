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
 *
 * This sample demonstrates implements matrix multiplication which makes use of
 * shared memory to ensure data reuse, the matrix multiplication is done using
 * tiling approach.
 * With compute capability 8.0 or higher the CUDA kernels involved uses
 * asynchronously copy data from global to shared memory; a.k.a., async-copy.
 * This sample has been written for clarity of exposition to illustrate various
 * CUDA programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda/pipeline>

#if __CUDA_ARCH__ >= 700
#include <cuda/barrier>
#endif
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

enum kernels {
  AsyncCopyMultiStageLargeChunk = 0,
  AsyncCopyLargeChunk = 1,
  AsyncCopyLargeChunkAWBarrier = 2,
  AsyncCopyMultiStageSharedState = 3,
  AsyncCopyMultiStage = 4,
  AsyncCopySingleStage = 5,
  Naive = 6,
  NaiveLargeChunk = 7
};

const char *kernelNames[] = {"AsyncCopyMultiStageLargeChunk",
                             "AsyncCopyLargeChunk",
                             "AsyncCopyLargeChunkAWBarrier",
                             "AsyncCopyMultiStageSharedState",
                             "AsyncCopyMultiStage",
                             "AsyncCopySingleStage",
                             "Naive",
                             "NaiveLargeChunk"};

constexpr int blockSize = 16;

// Multi Stage memcpy_async pipeline with large chunk copy
template <int BLOCK_SIZE>
__global__ void MatrixMulAsyncCopyMultiStageLargeChunk(
    float *__restrict__ C, const float *__restrict__ A,
    const float *__restrict__ B, int wA, int wB) {
  // Requires BLOCK_SIZE % 4 == 0

  // Multi-stage pipeline version
  constexpr size_t maxPipelineStages = 4;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A for each stage
  __shared__ alignas(
      alignof(float4)) float As[maxPipelineStages][BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B for each stage
  __shared__ alignas(
      alignof(float4)) float Bs[maxPipelineStages][BLOCK_SIZE][BLOCK_SIZE];

  float Csub = 0.0;

  // Index of the first sub-matrix of A processed by the block
  const int aBegin = wA * (BLOCK_SIZE)*blockIdx.y;

  // Index of the last sub-matrix of A processed by the block
  const int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  const int bBegin = BLOCK_SIZE * blockIdx.x;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;

  const int t4x = threadIdx.x * 4;
  const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin, i = 0, aStage = aBegin, bStage = bBegin,
           iStage = 0;
       a <= aEnd; a += aStep, b += bStep, ++i) {
    // Load the matrices from device memory to shared memory; each thread loads
    // one element of each matrix
    for (; aStage <= a + aStep * maxPipelineStages;
         aStage += aStep, bStage += bStep, ++iStage) {
      pipe.producer_acquire();
      if (aStage <= aEnd && t4x < BLOCK_SIZE) {
        // Rotating buffer
        const int j = iStage % maxPipelineStages;
        cuda::memcpy_async(&As[j][threadIdx.y][t4x],
                           &A[aStage + wA * threadIdx.y + t4x], shape4, pipe);
        cuda::memcpy_async(&Bs[j][threadIdx.y][t4x],
                           &B[aStage + wA * threadIdx.y + t4x], shape4, pipe);
      }
      pipe.producer_commit();
    }

    pipe.consumer_wait();
    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Rotating buffer
    const int j = i % maxPipelineStages;

// Multiply the two matrices together;
// each thread computes one element
// of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[j][threadIdx.y][k] * Bs[j][k][threadIdx.x];
    }
    pipe.consumer_release();

    // Don't have to synchronize because maxPipelineStages is greater than one
    // therefore next iteration is loading to a different buffer.
  }

  // Write the block sub-matrix to device memory;
  // each thread writes four element
  int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
  C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

// Single Stage memcpy_async pipeline with Large copy chunk (float4)
template <int BLOCK_SIZE>
__global__ void MatrixMulAsyncCopyLargeChunk(float *__restrict__ C,
                                             const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             int wA, int wB) {
  // Requires BLOCK_SIZE % 4 == 0

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ alignas(alignof(float4)) float As[BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ alignas(alignof(float4)) float Bs[BLOCK_SIZE][BLOCK_SIZE];

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * blockIdx.y;

  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * blockIdx.x;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;

  // Single-stage pipeline version
  float Csub = 0.0;

  const int t4x = threadIdx.x * 4;
  const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Load the matrices from device memory to shared memory;
    // a subset of threads loads a contiguous chunk of elements.

    // Previously, per-thread:
    // As[ty][tx] = A[a + wA * ty + tx];
    // Bs[ty][tx] = B[b + wB * ty + tx];

    // Now, one fourth of the threads load four elements of each matrix
    if (t4x < BLOCK_SIZE) {
      pipe.producer_acquire();

      cuda::memcpy_async(&As[threadIdx.y][t4x], &A[a + wA * threadIdx.y + t4x],
                         shape4, pipe);
      cuda::memcpy_async(&Bs[threadIdx.y][t4x], &B[a + wA * threadIdx.y + t4x],
                         shape4, pipe);

      pipe.producer_commit();
      pipe.consumer_wait();
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

// Multiply the two matrices together;
// each thread computes one element
// of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    pipe.consumer_release();

    // Synchronize to make sure that the preceding
    // computation is done before overwriting the
    // shared memory sub-matrix buffers As and Bs in the next iteration.
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes four element
  int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
  C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

// Single Stage memcpy_async pipeline with Large copy chunk (float4) using
// arrive-wait barrier
template <int BLOCK_SIZE>
__global__ void MatrixMulAsyncCopyLargeChunkAWBarrier(
    float *__restrict__ C, const float *__restrict__ A,
    const float *__restrict__ B, int wA, int wB) {
#if __CUDA_ARCH__ >= 700
#pragma diag_suppress static_var_with_dynamic_init
  // Requires BLOCK_SIZE % 4 == 0

  __shared__ cuda::barrier<cuda::thread_scope_block> bar;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ alignas(alignof(float4)) float As[BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ alignas(alignof(float4)) float Bs[BLOCK_SIZE][BLOCK_SIZE];

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x * blockDim.y);
  }
  __syncthreads();

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * blockIdx.y;

  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * blockIdx.x;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;

  float Csub = 0.0;

  const int t4x = threadIdx.x * 4;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Load the matrices from device memory to shared memory;
    // a subset of threads loads a contiguous chunk of elements.

    // Now, one fourth of the threads load four elements of each matrix
    if (t4x < BLOCK_SIZE) {
      float4 *const A4s = reinterpret_cast<float4 *>(&As[threadIdx.y][t4x]);
      float4 *const B4s = reinterpret_cast<float4 *>(&Bs[threadIdx.y][t4x]);
      const float4 *const A4 =
          reinterpret_cast<const float4 *>(&A[a + wA * threadIdx.y + t4x]);
      const float4 *const B4 =
          reinterpret_cast<const float4 *>(&B[a + wA * threadIdx.y + t4x]);

      cuda::memcpy_async(A4s, A4, sizeof(float4), bar);
      cuda::memcpy_async(B4s, B4, sizeof(float4), bar);
    }

    // Synchronize to make sure the matrices are loaded
    bar.arrive_and_wait();

// Multiply the two matrices together;
// each thread computes one element
// of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    // Synchronize to make sure that the preceding
    // computation is done before overwriting the
    // shared memory sub-matrix buffers As and Bs in the next iteration.
    bar.arrive_and_wait();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes four element
  int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
  C[c + wB * threadIdx.y + threadIdx.x] = Csub;
#endif
}

// Single Stage memcpy_async pipeline with float copy
template <int BLOCK_SIZE>
__global__ void MatrixMulAsyncCopySingleStage(float *C, const float *A,
                                              const float *B, int wA, int wB) {
  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * blockIdx.y;

  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * blockIdx.x;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;

  // Single-stage pipeline version
  float Csub = 0.0;

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape1 = cuda::aligned_size_t<alignof(float)>(sizeof(float));

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Load the matrices from device memory to shared memory; each thread loads
    // one element of each matrix
    {
      pipe.producer_acquire();

      cuda::memcpy_async(&As[threadIdx.y][threadIdx.x],
                         &A[a + wA * threadIdx.y + threadIdx.x], shape1, pipe);
      cuda::memcpy_async(&Bs[threadIdx.y][threadIdx.x],
                         &B[b + wB * threadIdx.y + threadIdx.x], shape1, pipe);

      pipe.producer_commit();
    }

    pipe.consumer_wait();
    // Synchronize to make sure the matrices are loaded
    __syncthreads();

// Multiply the two matrices together;
// each thread computes one element
// of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    // Synchronize to make sure that the preceding
    // computation is done before overwriting the
    // shared memory sub-matrix buffers As and Bs in the next iteration.
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes four element
  int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
  C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

// Multi Stage memcpy_async thread_scope_thread pipeline with single-element
// async-copy
template <int BLOCK_SIZE>
__global__ void MatrixMulAsyncCopyMultiStage(float *__restrict__ C,
                                             const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             int wA, int wB) {
  // Multi-stage pipeline version
  constexpr size_t maxPipelineStages = 4;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A for each stage
  __shared__ float As[maxPipelineStages][BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B for each stage
  __shared__ float Bs[maxPipelineStages][BLOCK_SIZE][BLOCK_SIZE];

  float Csub = 0.0;

  // Index of the first sub-matrix of A processed by the block
  const int aBegin = wA * BLOCK_SIZE * blockIdx.y;

  // Index of the last sub-matrix of A processed by the block
  const int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  const int bBegin = BLOCK_SIZE * blockIdx.x;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape1 = cuda::aligned_size_t<alignof(float)>(sizeof(float));

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin, i = 0, aStage = aBegin, bStage = bBegin,
           iStage = 0;
       a <= aEnd; a += aStep, b += bStep, ++i) {
    // Load the matrices from device memory to shared memory; each thread loads
    // one element of each matrix

    for (; aStage <= a + aStep * maxPipelineStages;
         aStage += aStep, bStage += bStep, ++iStage) {
      if (aStage <= aEnd) {
        // Rotating buffer
        const int j = iStage % maxPipelineStages;

        pipe.producer_acquire();

        cuda::memcpy_async(&As[j][threadIdx.y][threadIdx.x],
                           &A[aStage + wA * threadIdx.y + threadIdx.x], shape1,
                           pipe);
        cuda::memcpy_async(&Bs[j][threadIdx.y][threadIdx.x],
                           &B[bStage + wB * threadIdx.y + threadIdx.x], shape1,
                           pipe);

        pipe.producer_commit();
      }
    }
    pipe.consumer_wait();

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    const int j = i % maxPipelineStages;

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[j][threadIdx.y][k] * Bs[j][k][threadIdx.x];
    }

    pipe.consumer_release();
    // Don't have to synchronize because maxPipelineStages is greater than one
    // therefore next iteration is loading to a different buffer.
  }

  // Write the block sub-matrix to device memory;
  // each thread writes four element
  int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
  C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

// Multi Stage shared state memcpy_async pipeline thread_scope_block
// with parititioned producer & consumer, here we've 1 warp as producer
// group which issues memcpy_async operations and rest all warps are part of
// consumer group which perform gemm computation on the loaded matrices by
// producer.
template <int BLOCK_SIZE_X>
__global__ void MatrixMulAsyncCopyMultiStageSharedState(
    float *__restrict__ C, const float *__restrict__ A,
    const float *__restrict__ B, int wA, int wB) {
  // Multi-stage pipeline version
  constexpr size_t maxPipelineStages = 4;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A for each stage
  __shared__ float As[maxPipelineStages][BLOCK_SIZE_X][BLOCK_SIZE_X];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B for each stage
  __shared__ float Bs[maxPipelineStages][BLOCK_SIZE_X][BLOCK_SIZE_X];

  float Csub = 0.0;

  // Index of the first sub-matrix of A processed by the block
  const int aBegin = wA * BLOCK_SIZE_X * blockIdx.y;

  // Index of the last sub-matrix of A processed by the block
  const int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  constexpr int aStep = BLOCK_SIZE_X;

  // Index of the first sub-matrix of B processed by the block
  const int bBegin = BLOCK_SIZE_X * blockIdx.x;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE_X * wB;

  auto cta = cg::this_thread_block();

  const auto shape1 = cuda::aligned_size_t<alignof(float)>(sizeof(float));
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block,
                                         maxPipelineStages> shared_state;
  constexpr int consumer_row_count = BLOCK_SIZE_X;

  const auto thread_role = (cta.thread_index().y < consumer_row_count)
                               ? cuda::pipeline_role::consumer
                               : cuda::pipeline_role::producer;
  auto pipe = cuda::make_pipeline(cta, &shared_state, thread_role);

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin, i = 0, aStage = aBegin, bStage = bBegin,
           iStage = 0;
       a <= aEnd; a += aStep, b += bStep, ++i) {
    if (threadIdx.y >= consumer_row_count) {
      // this is a whole producer warp because threadIdx.y >= 16 where 16 ==
      // consumer_row_count,
      // which loads the matrices from device memory to shared memory;
      for (; aStage <= a + aStep * maxPipelineStages;
           aStage += aStep, bStage += bStep, ++iStage) {
        if (aStage <= aEnd) {
          // Rotating buffer
          const int j = iStage % maxPipelineStages;
          const int strideRows = (blockDim.y - consumer_row_count);
          pipe.producer_acquire();
          for (int rowId = threadIdx.y - consumer_row_count;
               rowId < BLOCK_SIZE_X; rowId += strideRows) {
            cuda::memcpy_async(&As[j][rowId][threadIdx.x],
                               &A[aStage + wA * rowId + threadIdx.x], shape1,
                               pipe);
            cuda::memcpy_async(&Bs[j][rowId][threadIdx.x],
                               &B[bStage + wB * rowId + threadIdx.x], shape1,
                               pipe);
          }
          pipe.producer_commit();
        }
      }
    } else {
      // this is a whole set of consumer group because threadIdx.y <
      // consumer_row_count where consumer_row_count == 16,
      // which computes gemm operation on matrices loaded in shared memory by
      // producer warp.
      const int j = i % maxPipelineStages;
      // Synchronize consumer group to make sure the matrices are loaded by
      // producer group.
      pipe.consumer_wait();
// Multiply the two matrices together;
// each thread computes one element
// of the block sub-matrix
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE_X; ++k) {
        Csub += As[j][threadIdx.y][k] * Bs[j][k][threadIdx.x];
      }
      pipe.consumer_release();
    }
  }

  // Write the block sub-matrix to device memory;
  // each thread writes four element
  if (threadIdx.y < consumer_row_count) {
    const int c = wB * BLOCK_SIZE_X * blockIdx.y + BLOCK_SIZE_X * blockIdx.x;
    C[c + wB * threadIdx.y + threadIdx.x] = Csub;
  }
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE>
__global__ void MatrixMulNaive(float *C, float *A, float *B, int wA, int wB) {
  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * blockIdx.y;

  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * blockIdx.x;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[threadIdx.y][threadIdx.x] = A[a + wA * threadIdx.y + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[b + wB * threadIdx.y + threadIdx.x];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

// Multiply the two matrices together;
// each thread computes one element
// of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
  C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

template <int BLOCK_SIZE>
__global__ void MatrixMulNaiveLargeChunk(float *C, float *A, float *B, int wA,
                                         int wB) {
  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ alignas(alignof(float4)) float As[BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ alignas(alignof(float4)) float Bs[BLOCK_SIZE][BLOCK_SIZE];

  int t4x = threadIdx.x * 4;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * blockIdx.y;

  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * blockIdx.x;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Load the matrices from device memory
    // to shared memory;

    // One fourth of the threads load four elements of each matrix
    if (t4x < BLOCK_SIZE) {
      float4 *const A4s = reinterpret_cast<float4 *>(&As[threadIdx.y][t4x]);
      float4 *const B4s = reinterpret_cast<float4 *>(&Bs[threadIdx.y][t4x]);
      const float4 *const A4 =
          reinterpret_cast<float4 *>(&A[a + wA * threadIdx.y + t4x]);
      const float4 *const B4 =
          reinterpret_cast<float4 *>(&B[a + wA * threadIdx.y + t4x]);
      *A4s = *A4;
      *B4s = *B4;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

// Multiply the two matrices together;
// each thread computes one element
// of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
  C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

void ConstantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

/**
 * Run matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char **argv, const dim3 &dimsA, const dim3 &dimsB,
                   kernels kernel_number) {
  // Allocate host memory for matrices A and B
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B;
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  cudaStream_t stream;

  // Initialize host memory
  const float valB = 2.10f;
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, valB);

  // Allocate device memory
  float *d_A, *d_B, *d_C;

  // Allocate host matrix C
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float *h_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // copy host memory to device
  checkCudaErrors(
      cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(
      cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemsetAsync(d_C, 0, mem_size_C, stream));

  // Setup execution parameters
  dim3 threads(blockSize, blockSize);
  dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

  // Here the block size is 16x18, where first 16 rows are consumer thread group
  // and last 2 rows (1 warp) is producer thread group
  dim3 threadsSharedStateKernel(blockSize, blockSize + 2, 1);
  dim3 gridSharedStateKernel(dimsB.x / threadsSharedStateKernel.x,
                             dimsA.y / threadsSharedStateKernel.x);

  printf("Running kernel = %d - %s\n", kernel_number,
         kernelNames[kernel_number]);
  // Create and start timer
  printf("Computing result using CUDA Kernel...\n");

  // Performs warmup operation using matrixMul CUDA kernel
  switch (kernel_number) {
    case AsyncCopyMultiStageLargeChunk:
    default:
      MatrixMulAsyncCopyMultiStageLargeChunk<
          blockSize><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x,
                                                   dimsB.x);
      break;
    case AsyncCopyLargeChunk:
      MatrixMulAsyncCopyLargeChunk<blockSize><<<grid, threads, 0, stream>>>(
          d_C, d_A, d_B, dimsA.x, dimsB.x);
      break;
    case AsyncCopyLargeChunkAWBarrier:
      MatrixMulAsyncCopyLargeChunkAWBarrier<
          blockSize><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x,
                                                   dimsB.x);
      break;
    case AsyncCopyMultiStageSharedState:
      MatrixMulAsyncCopyMultiStageSharedState<blockSize><<<
          gridSharedStateKernel, threadsSharedStateKernel, 0, stream>>>(
          d_C, d_A, d_B, dimsA.x, dimsB.x);
      break;
    case AsyncCopyMultiStage:
      MatrixMulAsyncCopyMultiStage<blockSize><<<grid, threads, 0, stream>>>(
          d_C, d_A, d_B, dimsA.x, dimsB.x);
      break;
    case AsyncCopySingleStage:
      MatrixMulAsyncCopySingleStage<blockSize><<<grid, threads, 0, stream>>>(
          d_C, d_A, d_B, dimsA.x, dimsB.x);
      break;
    case Naive:
      MatrixMulNaive<blockSize><<<grid, threads, 0, stream>>>(d_C, d_A, d_B,
                                                              dimsA.x, dimsB.x);
      break;
    case NaiveLargeChunk:
      MatrixMulNaiveLargeChunk<blockSize><<<grid, threads, 0, stream>>>(
          d_C, d_A, d_B, dimsA.x, dimsB.x);
      break;
  }

  printf("done\n");
  checkCudaErrors(cudaStreamSynchronize(stream));

  // Execute the kernel
  int nIter = 100;

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));

  for (int j = 0; j < nIter; j++) {
    switch (kernel_number) {
      case AsyncCopyMultiStageLargeChunk:
      default:
        MatrixMulAsyncCopyMultiStageLargeChunk<
            blockSize><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x,
                                                     dimsB.x);
        break;
      case AsyncCopyLargeChunk:
        MatrixMulAsyncCopyLargeChunk<blockSize><<<grid, threads, 0, stream>>>(
            d_C, d_A, d_B, dimsA.x, dimsB.x);
        break;
      case AsyncCopyLargeChunkAWBarrier:
        MatrixMulAsyncCopyLargeChunkAWBarrier<
            blockSize><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x,
                                                     dimsB.x);
        break;
      case AsyncCopyMultiStageSharedState:
        MatrixMulAsyncCopyMultiStageSharedState<blockSize><<<
            gridSharedStateKernel, threadsSharedStateKernel, 0, stream>>>(
            d_C, d_A, d_B, dimsA.x, dimsB.x);
        break;
      case AsyncCopyMultiStage:
        MatrixMulAsyncCopyMultiStage<blockSize><<<grid, threads, 0, stream>>>(
            d_C, d_A, d_B, dimsA.x, dimsB.x);
        break;
      case AsyncCopySingleStage:
        MatrixMulAsyncCopySingleStage<blockSize><<<grid, threads, 0, stream>>>(
            d_C, d_A, d_B, dimsA.x, dimsB.x);
        break;
      case Naive:
        MatrixMulNaive<blockSize><<<grid, threads, 0, stream>>>(
            d_C, d_A, d_B, dimsA.x, dimsB.x);
        break;
      case NaiveLargeChunk:
        MatrixMulNaiveLargeChunk<blockSize><<<grid, threads, 0, stream>>>(
            d_C, d_A, d_B, dimsA.x, dimsB.x);
        break;
    }
  }

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                             static_cast<double>(dimsA.y) *
                             static_cast<double>(dimsB.x);
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
      " WorkgroupSize= %u threads/block\n",
      gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

  // Copy result from device to host
  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  printf("Checking computed result for correctness: ");
  bool correct = true;

  // test relative error by the formula
  // |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-6;  // machine zero

  for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
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

  // Clean up memory
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  printf(
      "\nNOTE: The CUDA Samples are not meant for performance "
      "measurements. Results may vary when GPU Boost is enabled.\n");

  if (correct) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}

int main(int argc, char **argv) {
  printf("[globalToShmemAsyncCopy] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
    printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
    printf(
        "      -kernel=kernel_number (0 - AsyncCopyMultiStageLargeChunk; 1 - "
        "AsyncCopyLargeChunk)\n");
    printf(
        "                            (2 - AsyncCopyLargeChunkAWBarrier; 3 - "
        "AsyncCopyMultiStageSharedState)\n");
    printf(
        "                            (4 - AsyncCopyMultiStage; 5 - "
        "AsyncCopySingleStage; 6 - Naive without memcpy_async)\n");
    printf(
        "                            (7 - NaiveLargeChunk without "
        "memcpy_async)\n");
    printf(
        "  Note: Outer matrix dimensions of A & B matrices must be equal.\n");

    exit(EXIT_SUCCESS);
  }

  // This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line
  int dev = findCudaDevice(argc, (const char **)argv);

  int matrixBlock = 32;
  dim3 dimsA(10 * 4 * matrixBlock, 10 * 4 * matrixBlock, 1);
  dim3 dimsB(10 * 4 * matrixBlock, 10 * 4 * matrixBlock, 1);

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

  kernels selected_kernel = AsyncCopyMultiStageLargeChunk;

  // kernel to run - default (AsyncCopyMultiStageLargeChunk == 0)
  if (checkCmdLineFlag(argc, (const char **)argv, "kernel")) {
    int kernel_number =
        getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
    if (kernel_number < 8) {
      selected_kernel = (kernels)kernel_number;
    } else {
      printf(
          "Error: kernel number should be between 0 to 6, you have entered "
          "%d\n",
          kernel_number);
      exit(EXIT_FAILURE);
    }
  }

  int major = 0;
  checkCudaErrors(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  if (major < 7) {
    printf("globalToShmemAsyncCopy requires SM 7.0 or higher.  Exiting...\n");
    exit(EXIT_WAIVED);
  }

  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x,
         dimsB.y);

  int matrix_result = MatrixMultiply(argc, argv, dimsA, dimsB, selected_kernel);

  exit(matrix_result);
}
