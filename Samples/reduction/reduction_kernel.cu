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
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <class T>
__device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    mySum += __shfl_down_sync(mask, mySum, offset);
  }
  return mySum;
}

#if __CUDA_ARCH__ >= 800
// Specialize warpReduceFunc for int inputs to use __reduce_add_sync intrinsic
// when on SM 8.0 or higher
template <>
__device__ __forceinline__ int warpReduceSum<int>(unsigned int mask,
                                                  int mySum) {
  mySum = __reduce_add_sync(mask, mySum);
  return mySum;
}
#endif

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved
   inactivity means that no whole warps are active, which is also very
   inefficient */
template <class T>
__global__ void reduce0(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;

  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    // modulo arithmetic is slow!
    if ((tid % (2 * s)) == 0) {
      sdata[tid] += sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* This version uses contiguous threads, but its interleaved
   addressing results in many shared memory bank conflicts.
*/
template <class T>
__global__ void reduce1(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;

  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;

    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
template <class T>
__global__ void reduce2(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;

  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory.
*/
template <class T>
__global__ void reduce3(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  T mySum = (i < n) ? g_idata[i] : 0;

  if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = mySum = mySum + sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = mySum;
}

/*
    This version uses the warp shuffle operation if available to reduce
    warp synchronization. When shuffle is not available the final warp's
    worth of work is unrolled to reduce looping overhead.

    See
   http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
    for additional information about using shuffle to perform a reduction
    within a warp.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void reduce4(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  T mySum = (i < n) ? g_idata[i] : 0;

  if (i + blockSize < n) mySum += g_idata[i + blockSize];

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] = mySum = mySum + sdata[tid + s];
    }

    cg::sync(cta);
  }

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum += tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

/*
    This version is completely unrolled, unless warp shuffle is available, then
    shuffle is used within a loop.  It uses a template parameter to achieve
    optimal code for any (power of 2) number of threads.  This requires a switch
    statement in the host code to handle all the different thread block sizes at
    compile time. When shuffle is available, it is used to reduce warp
   synchronization.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void reduce5(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

  T mySum = (i < n) ? g_idata[i] : 0;

  if (i + blockSize < n) mySum += g_idata[i + blockSize];

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = mySum = mySum + sdata[tid + 64];
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum += tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

/*
    This version adds multiple elements per thread sequentially.  This reduces
   the overall cost of the algorithm while keeping the work complexity O(n) and
   the step complexity O(log n). (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      mySum += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        mySum += g_idata[i + blockSize];
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      mySum += g_idata[i];
      i += gridSize;
    }
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = mySum = mySum + sdata[tid + 64];
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum += tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce7(const T *__restrict__ g_idata, T *__restrict__ g_odata,
                        unsigned int n) {
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;
  unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
  maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
  const unsigned int mask = (0xffffffff) >> maskLength;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      mySum += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        mySum += g_idata[i + blockSize];
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      mySum += g_idata[i];
      i += gridSize;
    }
  }

  // Reduce within warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
  // SM 8.0
  mySum = warpReduceSum<T>(mask, mySum);

  // each thread puts its local sum into shared memory
  if ((tid % warpSize) == 0) {
    sdata[tid / warpSize] = mySum;
  }

  __syncthreads();

  const unsigned int shmem_extent =
      (blockSize / warpSize) > 0 ? (blockSize / warpSize) : 1;
  const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
  if (tid < shmem_extent) {
    mySum = sdata[tid];
    // Reduce final warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
    // SM 8.0
    mySum = warpReduceSum<T>(ballot_result, mySum);
  }

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = mySum;
  }
}

// Performs a reduction step and updates numTotal with how many are remaining
template <typename T, typename Group>
__device__ T cg_reduce_n(T in, Group &threads) {
  return cg::reduce(threads, in, cg::plus<T>());
}

template <class T>
__global__ void cg_reduce(T *g_idata, T *g_odata, unsigned int n) {
  // Shared memory for intermediate steps
  T *sdata = SharedMemory<T>();
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Handle to tile in thread block
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

  unsigned int ctaSize = cta.size();
  unsigned int numCtas = gridDim.x;
  unsigned int threadRank = cta.thread_rank();
  unsigned int threadIndex = (blockIdx.x * ctaSize) + threadRank;

  T threadVal = 0;
  {
    unsigned int i = threadIndex;
    unsigned int indexStride = (numCtas * ctaSize);
    while (i < n) {
      threadVal += g_idata[i];
      i += indexStride;
    }
    sdata[threadRank] = threadVal;
  }

  // Wait for all tiles to finish and reduce within CTA
  {
    unsigned int ctaSteps = tile.meta_group_size();
    unsigned int ctaIndex = ctaSize >> 1;
    while (ctaIndex >= 32) {
      cta.sync();
      if (threadRank < ctaIndex) {
        threadVal += sdata[threadRank + ctaIndex];
        sdata[threadRank] = threadVal;
      }
      ctaSteps >>= 1;
      ctaIndex >>= 1;
    }
  }

  // Shuffle redux instead of smem redux
  {
    cta.sync();
    if (tile.meta_group_rank() == 0) {
      threadVal = cg_reduce_n(threadVal, tile);
    }
  }

  if (threadRank == 0) g_odata[blockIdx.x] = threadVal;
}

extern "C" bool isPow2(unsigned int x);

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void reduce(int size, int threads, int blocks, int whichKernel, T *d_idata,
            T *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

  // choose which of the optimized versions of reduction to launch
  switch (whichKernel) {
    case 0:
      reduce0<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 1:
      reduce1<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 2:
      reduce2<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 3:
      reduce3<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 4:
      switch (threads) {
        case 512:
          reduce4<T, 512>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 256:
          reduce4<T, 256>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 128:
          reduce4<T, 128>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 64:
          reduce4<T, 64>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 32:
          reduce4<T, 32>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 16:
          reduce4<T, 16>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 8:
          reduce4<T, 8>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 4:
          reduce4<T, 4>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 2:
          reduce4<T, 2>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 1:
          reduce4<T, 1>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;
      }

      break;

    case 5:
      switch (threads) {
        case 512:
          reduce5<T, 512>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 256:
          reduce5<T, 256>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 128:
          reduce5<T, 128>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 64:
          reduce5<T, 64>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 32:
          reduce5<T, 32>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 16:
          reduce5<T, 16>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 8:
          reduce5<T, 8>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 4:
          reduce5<T, 4>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 2:
          reduce5<T, 2>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 1:
          reduce5<T, 1>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;
      }

      break;

    case 6:
      if (isPow2(size)) {
        switch (threads) {
          case 512:
            reduce6<T, 512, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 256:
            reduce6<T, 256, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 128:
            reduce6<T, 128, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 64:
            reduce6<T, 64, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 32:
            reduce6<T, 32, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 16:
            reduce6<T, 16, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 8:
            reduce6<T, 8, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 4:
            reduce6<T, 4, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 2:
            reduce6<T, 2, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 1:
            reduce6<T, 1, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;
        }
      } else {
        switch (threads) {
          case 512:
            reduce6<T, 512, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 256:
            reduce6<T, 256, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 128:
            reduce6<T, 128, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 64:
            reduce6<T, 64, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 32:
            reduce6<T, 32, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 16:
            reduce6<T, 16, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 8:
            reduce6<T, 8, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 4:
            reduce6<T, 4, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 2:
            reduce6<T, 2, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 1:
            reduce6<T, 1, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;
        }
      }

      break;

    case 7:
      // For reduce7 kernel we require only blockSize/warpSize
      // number of elements in shared memory
      smemSize = ((threads / 32) + 1) * sizeof(T);
      if (isPow2(size)) {
        switch (threads) {
          case 512:
            reduce7<T, 512, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 256:
            reduce7<T, 256, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 128:
            reduce7<T, 128, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 64:
            reduce7<T, 64, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 32:
            reduce7<T, 32, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 16:
            reduce7<T, 16, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 8:
            reduce7<T, 8, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 4:
            reduce7<T, 4, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 2:
            reduce7<T, 2, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 1:
            reduce7<T, 1, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;
        }
      } else {
        switch (threads) {
          case 512:
            reduce7<T, 512, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 256:
            reduce7<T, 256, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 128:
            reduce7<T, 128, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 64:
            reduce7<T, 64, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 32:
            reduce7<T, 32, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 16:
            reduce7<T, 16, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 8:
            reduce7<T, 8, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 4:
            reduce7<T, 4, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 2:
            reduce7<T, 2, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 1:
            reduce7<T, 1, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;
        }
      }

      break;
    case 8:
    default:
      cg_reduce<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;
  }
}

// Instantiate the reduction function for 3 types
template void reduce<int>(int size, int threads, int blocks, int whichKernel,
                          int *d_idata, int *d_odata);

template void reduce<float>(int size, int threads, int blocks, int whichKernel,
                            float *d_idata, float *d_odata);

template void reduce<double>(int size, int threads, int blocks, int whichKernel,
                             double *d_idata, double *d_odata);

#endif  // #ifndef _REDUCE_KERNEL_H_
