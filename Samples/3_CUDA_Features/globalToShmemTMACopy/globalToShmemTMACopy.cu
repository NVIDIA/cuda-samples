/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
 * This sample demonstrates how to:
 *
 * - Create a TensorMap (TMA descriptor)
 * - Load a 2D tile of data into shared memory
 *
 * Compile with:
 *
 * nvcc -arch sm_90 globalToShmemTMACopy.cu -o globalToShmemTMACopy
 *
 * It can be that the compiler issues the following note. This can be safely ignored.
 *
 *   note: the ABI for passing parameters with 64-byte alignment has changed in
 *   GCC 4.6
 *
 */
#include <cstdio>                      // fprintf
#include <vector>                      // std::vector

#include <cuda.h>                      // CUtensorMap
#include <cuda_awbarrier_primitives.h> // __mbarrier_*

#include "util.h"                      // CUDA_CHECK macro

/*
 * Constants.
 */
constexpr size_t W_global = 1024; // Width of tensor (in # elements)
constexpr size_t H_global = 1024; // Height of tensor (in # elements)

constexpr int SMEM_W = 32;     // Width of shared memory buffer (in # elements)
constexpr int SMEM_H = 8;      // Height of shared memory buffer (in # elements)

/*
 * CUDA Driver API
 */

// The type of the cuTensorMapEncodeTiled function.
using cuTensorMapEncodeTiled_t = decltype(cuTensorMapEncodeTiled);

// Get function pointer to driver API cuTensorMapEncodeTiled
cuTensorMapEncodeTiled_t * get_cuTensorMapEncodeTiled() {
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
  void* cuda_ptr = nullptr;
  unsigned long long flags = cudaEnableDefault;

  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER__ENTRY__POINT.html
  CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &cuda_ptr, flags));

  return reinterpret_cast<cuTensorMapEncodeTiled_t*>(cuda_ptr);
}

/*
 * PTX wrappers
 */

inline __device__ __mbarrier_token_t barrier_arrive1_tx(
  __mbarrier_t *barrier, uint32_t expected_tx_count
)
{
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
  __mbarrier_token_t token;

  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
               : "=l"(token)
               : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))), "r"(expected_tx_count)
               : "memory");
  return token;
}

inline __device__ bool barrier_try_wait_token(__mbarrier_t *barrier, __mbarrier_token_t token)
{
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
  //
  // This function returns a bool, so that software can retry.
  //
  //  The HW only provides best-effort waiting support. The wait time is limited
  //  by the HW capability, after which a fail occurs, in which case the SW is
  //  responsible for retrying.
  int __ready;
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
               "selp.b32 %0, 1, 0, p;\n\t"
               "}"
               : "=r"(__ready)
               : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))),
                 "l"(token)
               : "memory");
  return __ready;
}

inline __device__ void cp_async_bulk_tensor_2d(
  __mbarrier_t *barrier, void *dst, int access_coord_x, int access_coord_y, const CUtensorMap *tensor_desc)
{
  unsigned smem_int_ptr = static_cast<unsigned int>(__cvta_generic_to_shared(dst));
  unsigned smem_barrier_int_ptr = static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  uint64_t tensor_desc_ptr = reinterpret_cast<uint64_t>(tensor_desc);

  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
    "[%0], [%1, {%2, %3}], [%4];\n"
    :
    : "r"(smem_int_ptr),
      "l"(tensor_desc_ptr),
      "r"(access_coord_x),
      "r"(access_coord_y),
      "r"(smem_barrier_int_ptr)
    : "memory");
}

// Layout of shared memory. It contains:
//
// - a buffer to hold a subset of a tensor,
// - a shared memory barrier.
template <int H, int W>
struct smem_t {

  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  struct alignas(128) tensor_buffer {
    int data[H][W];

    __device__ constexpr int width() {return W;}
    __device__ constexpr int height() {return H;}
  };

  tensor_buffer buffer;

  // Put the barrier behind the tensor buffer to prevent 100+ bytes of padding.
  __mbarrier_t bar;

  __device__ constexpr int buffer_size_in_bytes() {
    return sizeof(tensor_buffer::data);
  }
};


/*
 * Main kernel: takes a TMA descriptor and two coordinates.
 *
 * Loads a tile into shared memory using TMA and prints the tile.
 *
 */
__global__ void kernel(const __grid_constant__ CUtensorMap tma_desc, int x_0, int y_0) {
  /*
   * ***NOTE***:

     A CUtensorMap can only be passed as a `const __grid_constant__`
     parameter. Passing a CUtensorMap in any other way from the host to
     device can result in difficult if not impossible to debug failures.

  */

  // Declare shared memory to hold tensor buffer and shared memory barrier.
  __shared__ smem_t<SMEM_H, SMEM_W> smem;

  // Utility variable to elect a leader thread.
  bool leader = threadIdx.x == 0;


  if (leader) {
    // Initialize barrier. We will participate in the barrier with `blockDim.x`
    // threads.
    __mbarrier_init(&smem.bar, blockDim.x);
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();


  // This token is created when arriving on the shared memory barrier. It is
  // used again when waiting on the barrier.
  __mbarrier_token_t token;

  // Load first  batch
  if (leader) {
    // Initiate bulk tensor copy.
    cp_async_bulk_tensor_2d(&smem.bar, &smem.buffer.data, x_0, y_0, &tma_desc);
    // Arrive with arrival count of 1 and expected transaction count equal to
    // the number of bytes that are copied by cp_async_bulk_tensor_2d.
    token = barrier_arrive1_tx(&smem.bar, smem.buffer_size_in_bytes());
  } else {
    // Other threads arrive with arrival count of 1 and expected tx count of 0.
    token = barrier_arrive1_tx(&smem.bar, 0);
  }

  // The barrier will flip when the following two conditions have been met:
  //
  // - Its arrival count reaches blockDim.x (see __mbarrier_init above).
  //   Typically, each thread will arrive with an arrival count of one so this
  //   indicates that all threads have arrived.
  //
  // - Its expected transaction count reaches smem.buffer_size_in_bytes(). The
  //   bulk tensor operation will increment the transaction count as it copies
  //   bytes.

  // Wait for barrier to flip. Try_wait puts the thread to sleep while waiting.
  // It is woken up when the barrier flips or when a hardware-defined number of
  // clock cycles have passed. In the second case, we retry waiting.
  while(! barrier_try_wait_token(&smem.bar, token)) { };

  // From this point onwards, the data in smem.buffer is readable by all threads
  // participating the in the barrier.

  // Print the data:
  if (leader) {
    printf("\n\nPrinting tile at coordinates x0 = %d, y0 = %d\n", x_0, y_0);

    // Print global x coordinates
    printf("global->\t");
    for (int x = 0; x < smem.buffer.width(); ++x) {
      printf("[%4d] ", x_0 + x);
    }
    printf("\n");

    // Print local x coordinates
    printf("local ->\t");
    for (int x = 0; x < smem.buffer.width(); ++x) {
      printf("[%4d] ", x);
    }
    printf("\n");

    for (int y = 0; y < smem.buffer.height(); ++y) {
      // Print global and local y coordinates
      printf("[%4d] [%2d]\t", y_0 + y, y);
      for (int x = 0; x < smem.buffer.width(); ++x) {
        printf(" %4d  ", smem.buffer.data[y][x]);
      }
      printf("\n");
    }

    // Invalidate barrier. If further computations were to take place in the
    // kernel, this allows the memory location of the shared memory barrier to
    // be repurposed.
    __mbarrier_inval(&smem.bar);
  }
}

int main(int argc, char **argv) {

  // Create a 2D tensor in GPU global memory containing linear indices 0, 1, 2, ... .
  // The data layout is row-major.

  // First fill in a vector on the host.
  std::vector<int> tensor_host(H_global * W_global);
  for (int i = 0; i < H_global * W_global; ++i) {
    tensor_host[i] = i;
  }

  // Move it to device
  int * tensor = nullptr;
  CUDA_CHECK(cudaMalloc(&tensor, H_global * W_global * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(tensor, tensor_host.data(), H_global * W_global * sizeof(int), cudaMemcpyHostToDevice));

  // Set up parameters to create TMA descriptor.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html

  CUtensorMap tma_desc{};
  CUtensorMapDataType dtype = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32;
  auto rank = 2;
  uint64_t size[rank] = {W_global, H_global};
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {W_global * sizeof(int)};
  // The box_size is the size of the shared memory buffer that is used as the destination of a TMA transfer.
  uint32_t box_size[rank] = {SMEM_W, SMEM_H};
  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};
  // Interleave patterns are sometimes used to accelerate loading of values that
  // are less than 4 bytes long.
  CUtensorMapInterleave interleave = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE;
  // Swizzling can be used to avoid shared memory bank conflicts.
  CUtensorMapSwizzle swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
  CUtensorMapL2promotion l2_promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE;
  // Any element that is outside of bounds will be set to zero by the TMA transfer.
  CUtensorMapFloatOOBfill oob_fill = CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  // Get a function pointer to the cuTensorMapEncodeTiled driver API.
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
      &tma_desc,    // CUtensorMap *tensorMap,
      dtype,        // CUtensorMapDataType tensorDataType,
      rank,         // cuuint32_t tensorRank,
      tensor,       // void *globalAddress,
      size,         // const cuuint64_t *globalDim,
      stride,       // const cuuint64_t *globalStrides,
      box_size,     // const cuuint32_t *boxDim,
      elem_stride,  // const cuuint32_t *elementStrides,
      interleave,   // CUtensorMapInterleave interleave,
      swizzle,      // CUtensorMapSwizzle swizzle,
      l2_promotion, // CUtensorMapL2promotion l2Promotion,
      oob_fill      // CUtensorMapFloatOOBfill oobFill);
    );
  // Print the result. Should be zero.
  printf("cuTensorMapEncodeTiled returned CUresult: %d\n\n", res);

  CUDA_CHECK(cudaDeviceSynchronize());

  dim3 grid(1);
  dim3 block(128);

  printf("Print the top right corner tile of the tensor:\n");
  kernel<<<grid, block>>>(tma_desc, 0, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Negative indices work:\n");
  kernel<<<grid, block>>>(tma_desc, -4, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("When the indices are out of bounds, the shared memory buffer is filled with zeros:\n");
  kernel<<<grid, block>>>(tma_desc, W_global, H_global);
  CUDA_CHECK(cudaDeviceSynchronize());

  printf(
    "\nCare must be taken to ensure that the coordinates result in a memory offset\n"
    "that is aligned to 16 bytes. With 32 bit integer elements, x coordinates\n"
    "that are not a multiple of 4 result in a non-recoverable error:\n"
  );
  kernel<<<grid, block>>>(tma_desc, 1, 0);
  CUDA_REPORT(cudaDeviceSynchronize());
  kernel<<<grid, block>>>(tma_desc, 2, 0);
  CUDA_REPORT(cudaDeviceSynchronize());
  kernel<<<grid, block>>>(tma_desc, 3, 0);
  CUDA_REPORT(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(tensor));
  return 0;
}
