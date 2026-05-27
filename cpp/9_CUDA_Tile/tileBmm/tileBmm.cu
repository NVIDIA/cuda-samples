/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
 * This sample demonstrates a static-persistent batched matrix multiplication
 * (BMM) using CUDA Tile C++. Given A of shape (Q, M, K) and B of shape
 * (Q, K, N), the kernel computes C = A x B of shape (Q, M, N). The grid
 * launches a fixed number of persistent blocks (sized from the device's SM
 * count); each block walks the (M, N, Q-chunk) tile space via a grid-stride
 * loop. The batch dimension is tiled by BLOCK_SIZE_Q so every iteration issues
 * a single rank-3 batched cuda::tiles::mma per K-step over tiles of shape
 * (BLOCK_SIZE_Q, BLOCK_SIZE_M, BLOCK_SIZE_K) x
 * (BLOCK_SIZE_Q, BLOCK_SIZE_K, BLOCK_SIZE_N).  Grouped ordering on
 * (pid_m, pid_n) gives L2 reuse. The accumulator is kept in float32 for
 * precision, and masked loads/stores handle tiles that overhang the matrix
 * or batch boundaries. Inputs and outputs use __half.
 *
 * A SIMT kernel is used to initialize the input matrices.
 */

#include "helper_cuda.h"

#include "cuda_tile.h"
#include "cuda_fp16.h"

#include <cstdio>
#include <cstdlib>

/* SIMT initializer for A (shape Q x M x K) and B (shape Q x K x N).
 * Values are bounded so the K-summed result fits comfortably in __half. */
__global__ void initializeMatrices(__half* a, __half* b,
                                   int Q, int M, int N, int K) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t a_size = std::size_t(Q) * M * K;
  std::size_t b_size = std::size_t(Q) * K * N;

  if (idx < a_size) {
    int k = idx % K;
    int m = (idx / K) % M;
    a[idx] = __half{float((m + k + 1) % 8) / 32.0f};
  }
  if (idx < b_size) {
    int n = idx % N;
    int k = (idx / N) % K;
    b[idx] = __half{float((k + n + 1) % 8) / 32.0f};
  }
}

/* Static-persistent tile kernel computing C = A @ B for batched 3D tensors.
 * A: (Q, M, K), B: (Q, K, N), C: (Q, M, N).  Both inputs are in their
 * natural (non-transposed) layout.  The grid is sized from the device SM
 * count and each block walks the (M, N, Q-chunk) tile space via a
 * grid-stride irange loop.  Each iteration consumes a chunk of BLOCK_SIZE_Q
 * batches with a single rank-3 batched mma per K-step. */
template<typename T, int BLOCK_SIZE_Q, int BLOCK_SIZE_M, int BLOCK_SIZE_N,
         int BLOCK_SIZE_K, int GROUP_SIZE_M, int Q, int M, int N, int K,
         int NUM_CTAS, int OCCUPANCY>
[[ using cutile :
    hint(0, num_cta_in_cga=NUM_CTAS),
    hint(0, occupancy=OCCUPANCY)
]]
__tile_global__ void persistent_bmm_kernel(const T* __restrict__ _a_ptr,
                                           const T* __restrict__ _b_ptr,
                                           T* __restrict__ _c_ptr) {
  namespace ct = cuda::tiles;

  /* tell the compiler the pointers are aligned (important for codegen) */
  const T* a_ptr = ct::assume_aligned<16>(_a_ptr);
  const T* b_ptr = ct::assume_aligned<16>(_b_ptr);
  T* c_ptr = ct::assume_aligned<16>(_c_ptr);

  /* accumulator tile kept in float32 for numerical precision; rank-3
   * so the batched mma can fold the (q, m, k) x (q, k, n) -> (q, m, n)
   * contraction in a single call. */
  using AccTile = ct::tile<float,
                           ct::shape<BLOCK_SIZE_Q, BLOCK_SIZE_M, BLOCK_SIZE_N>>;

  int bid = ct::bid().x;
  int num_programs = ct::num_blocks().x;

  /* tile counts include a chunked batch axis */
  constexpr int num_tiles_m = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
  constexpr int num_tiles_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
  constexpr int num_tiles_q = (Q + BLOCK_SIZE_Q - 1) / BLOCK_SIZE_Q;
  constexpr int total_tiles = num_tiles_m * num_tiles_n * num_tiles_q;

  /* loop-invariant partition views for A (Q, M, K), B (Q, K, N), C (Q, M, N) */
  auto a_layout = ct::layout_right_mapping{ct::extents{Q, M, K}};
  auto pA = ct::partition_view{
      ct::tensor_span{a_ptr, a_layout},
      ct::shape<BLOCK_SIZE_Q, BLOCK_SIZE_M, BLOCK_SIZE_K>{}};
  auto b_layout = ct::layout_right_mapping{ct::extents{Q, K, N}};
  auto pB = ct::partition_view{
      ct::tensor_span{b_ptr, b_layout},
      ct::shape<BLOCK_SIZE_Q, BLOCK_SIZE_K, BLOCK_SIZE_N>{}};
  auto c_layout = ct::layout_right_mapping{ct::extents{Q, M, N}};
  auto pC = ct::partition_view{
      ct::tensor_span{c_ptr, c_layout},
      ct::shape<BLOCK_SIZE_Q, BLOCK_SIZE_M, BLOCK_SIZE_N>{}};

  /* grid-stride loop over (pid_q_chunk, pid_m, pid_n) tiles */
  for (auto current_bid : ct::irange(bid, total_tiles, num_programs)) {
    /* decode the linear tile id with grouped ordering on (m, n) for L2 reuse */
    int pid_q = current_bid / (num_tiles_m * num_tiles_n);
    int num_pid_in_group = GROUP_SIZE_M * num_tiles_n;

    int current_bid_2d = current_bid % (num_tiles_m * num_tiles_n);
    int group_id = current_bid_2d / num_pid_in_group;
    int first_pid_m = group_id * GROUP_SIZE_M;
    int group_size_m_temp = num_tiles_m - first_pid_m;
    int group_size_m = (group_size_m_temp < GROUP_SIZE_M)
                       ? group_size_m_temp : GROUP_SIZE_M;
    int pid_m = first_pid_m + (current_bid_2d % group_size_m);
    int pid_n = (current_bid_2d % num_pid_in_group) / group_size_m;

    auto accumulator = ct::zeros<AccTile>();

    /* K-dimension accumulation loop; each iteration issues a single
     * rank-3 mma across BLOCK_SIZE_Q batches. */
    constexpr int num_k_tiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
    for (auto k_tile : ct::irange(0, num_k_tiles)) {
      auto a_tile = pA.load_masked(pid_q, pid_m, k_tile);
      auto b_tile = pB.load_masked(pid_q, k_tile, pid_n);
      accumulator = ct::mma(a_tile, b_tile, accumulator);
    }

    auto result = ct::element_cast<T>(accumulator);
    pC.store_masked(result, pid_q, pid_m, pid_n);
  }
}

int main() {
  /* tile-shape template parameters: multiples of 16 (tensor-core friendly)
   * that divide the test problem cleanly.  BLOCK_SIZE_Q controls how
   * many batches each block fuses into a single rank-3 mma.  NUM_CTAS and
   * OCCUPANCY are launch hints for the cutile compiler.  These values
   * mirror the production defaults used in the Ocean / TileGym BMM kernel
   * for sm_100-class GPUs. */
  constexpr int BLOCK_SIZE_Q = 1;
  constexpr int BLOCK_SIZE_M = 256;
  constexpr int BLOCK_SIZE_N = 256;
  constexpr int BLOCK_SIZE_K = 64;
  constexpr int GROUP_SIZE_M = 8;
  constexpr int NUM_CTAS     = 2;
  constexpr int OCCUPANCY    = 1;

  /* problem dimensions are compile-time NTTPs so partition extents fold and
   * total_tiles is constexpr inside the kernel.  Sizes are kept small so the
   * CPU reference comparison stays fast; the launch config above is still
   * the production sm_100 set (which is tuned for much larger shapes). */
  constexpr int Q = 4;
  constexpr int M = 256;
  constexpr int N = 256;
  constexpr int K = 128;

  std::size_t a_size = std::size_t(Q) * M * K;
  std::size_t b_size = std::size_t(Q) * K * N;
  std::size_t c_size = std::size_t(Q) * M * N;

  __half* d_A = nullptr;
  __half* d_B = nullptr;
  __half* d_C = nullptr;
  checkCudaErrors(cudaMalloc(&d_A, a_size * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_B, b_size * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_C, c_size * sizeof(__half)));

  /* populate A and B with deterministic test data on the device */
  int init_threads = 256;
  std::size_t init_elems = (a_size > b_size) ? a_size : b_size;
  int init_blocks = int((init_elems + init_threads - 1) / init_threads);
  initializeMatrices<<<init_blocks, init_threads>>>(d_A, d_B, Q, M, N, K);
  checkCudaErrors(cudaGetLastError());

  /* compute a CPU reference using double accumulation, then cast to __half */
  __half* h_A = new __half[a_size];
  __half* h_B = new __half[b_size];
  __half* h_C_ref = new __half[c_size];
  checkCudaErrors(cudaMemcpy(h_A, d_A, a_size * sizeof(__half), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_B, d_B, b_size * sizeof(__half), cudaMemcpyDeviceToHost));

  for (int q = 0; q < Q; ++q) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
          double av = double(float(h_A[(std::size_t(q) * M + m) * K + k]));
          double bv = double(float(h_B[(std::size_t(q) * K + k) * N + n]));
          acc += av * bv;
        }
        h_C_ref[(std::size_t(q) * M + m) * N + n] = __half{float(acc)};
      }
    }
  }

  /* launch the persistent BMM kernel: grid size mirrors the static-persistent
   * formula min(NUM_SMS / NUM_CTAS, total_tiles) * OCCUPANCY -- enough
   * blocks to either saturate the device or cover all tiles, whichever is
   * smaller. */
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
  int num_sms = prop.multiProcessorCount;

  constexpr int num_tiles_m = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
  constexpr int num_tiles_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
  constexpr int num_tiles_q = (Q + BLOCK_SIZE_Q - 1) / BLOCK_SIZE_Q;
  constexpr int total_tiles = num_tiles_m * num_tiles_n * num_tiles_q;

  int base_programs = num_sms / NUM_CTAS;
  int grid_size = (base_programs < total_tiles ? base_programs : total_tiles)
                  * OCCUPANCY;

  persistent_bmm_kernel<__half, BLOCK_SIZE_Q, BLOCK_SIZE_M, BLOCK_SIZE_N,
                        BLOCK_SIZE_K, GROUP_SIZE_M, Q, M, N, K,
                        NUM_CTAS, OCCUPANCY>
      <<<dim3(grid_size, 1, 1)>>>(d_A, d_B, d_C);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  __half* h_C = new __half[c_size];
  checkCudaErrors(cudaMemcpy(h_C, d_C, c_size * sizeof(__half), cudaMemcpyDeviceToHost));

  for (std::size_t idx = 0; idx < c_size; ++idx) {
    float got = float(h_C[idx]);
    float ref = float(h_C_ref[idx]);
    float diff = got > ref ? got - ref : ref - got;
    if (diff > 1e-1f) {
      printf("Expected: h_C[%zu] == %f\n", idx, ref);
      printf("Actual:   h_C[%zu] == %f\n", idx, got);

      return 1;
    }
  }

  printf("Success! BMM matches expected results.\n");

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_C_ref;
}
