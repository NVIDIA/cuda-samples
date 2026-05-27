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
 * This sample demonstrates matrix multiplication using CUDA Tile C++.
 * The input matrices are split into tiles. Each Tile block computes one
 * output tile by loading FP16 tiles from A and B, accumulating with
 * cuda::tiles::mma in FP32, and storing an FP32 result tile.
 *
 * The sample compares a naive implementation with an optimized implementation.
 * The optimized kernel adds compiler guidance such as pointer alignment and
 * divisibility assumptions, cuda::tiles::irange, and latency hints. The host
 * code validates both results against a CPU reference and uses CUDA events to
 * compare execution time.
 */

#include "helper_cuda.h"
#include "matmul_benchmark.h"

#include "cuda_tile.h"
#include "cuda_fp16.h"

#include <cstdlib>
#include <cstdio>
#include <vector>

constexpr int TILE_BLOCK_M = 32;
constexpr int TILE_BLOCK_N = 64;
constexpr int TILE_BLOCK_K = 64;
constexpr int LOAD_LATENCY = 5;
constexpr int STORE_LATENCY = 5;

/*
 * Baseline Tile C++ matmul. This keeps the same tensor_span, partition_view,
 * and ct::mma structure as the optimized kernel, but avoids optimization
 * hints, assumptions, Tile-specific loop guidance, and the divisibility
 * precondition used by non-masked view loads and stores.
 */
__tile_global__ void matmul_naive(float* C,
                                  const __half* A,
                                  const __half* B,
                                  int M, int N, int K) {

    namespace ct = cuda::tiles;

    // create tensor spans with runtime shapes (FP16 for A and B)
    auto a_span = ct::tensor_span{A, ct::extents{M, K}};
    auto b_span = ct::tensor_span{B, ct::extents{K, N}};
    auto c_span = ct::tensor_span{C, ct::extents{M, N}};
    
    // create partition views with compile-time tile sizes
    auto a_view = ct::partition_view{a_span, ct::shape<TILE_BLOCK_M, TILE_BLOCK_K>{}};
    auto b_view = ct::partition_view{b_span, ct::shape<TILE_BLOCK_K, TILE_BLOCK_N>{}};
    auto c_view = ct::partition_view{c_span, ct::shape<TILE_BLOCK_M, TILE_BLOCK_N>{}};
    
    // get block indices from the 2D grid
    auto [pid_m, pid_n, dummy] = ct::bid();
    
    // initialize FP32 accumulator
    auto acc = ct::zeros<ct::tile<float, ct::shape<TILE_BLOCK_M, TILE_BLOCK_N>>>();
    
    // loop over the K dimension in blocks
    int num_k_blocks = (K + TILE_BLOCK_K - 1) / TILE_BLOCK_K;
    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        ct::tile<__half, ct::shape<TILE_BLOCK_M, TILE_BLOCK_K>> a_tile;
        ct::tile<__half, ct::shape<TILE_BLOCK_K, TILE_BLOCK_N>> b_tile;
        
        // load blocks of A and B (FP16), using the default zero padding for boundary tiles
        a_tile = a_view.load_masked(pid_m, k_block);
        b_tile = b_view.load_masked(k_block, pid_n);
        
        // accumulate: acc += A_block @ B_block (FP16 inputs, FP32 accumulator)
        // ct::mma handles mixed precision: FP16 operands with FP32 accumulator.
        acc = ct::mma(a_tile, b_tile, acc);
    }
    
    // store result (FP32), suppressing stores outside the output matrix
    c_view.store_masked(acc, pid_m, pid_n);
}

/*
 * optimization strategy: declare non-aliasing pointers with __restrict__.
 * This gives the compiler more freedom to schedule loads and stores for A,
 * B, and C because the output tile cannot overlap the input tiles.
 */
__tile_global__ void matmul(float* __restrict__ _C,
                            const __half* __restrict__ _A,
                            const __half* __restrict__ _B,
                            int _M, int _N, int _K) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    /*
     * optimization strategy: communicate pointer alignment explicitly.
     * Assumptions are optimization facts, not runtime checks; the returned
     * value must be used, and violating the assumption is undefined behavior.
     */
    float* C = ct::assume_aligned(_C, 16_ic);
    const __half* A = ct::assume_aligned(_A, 16_ic);
    const __half* B = ct::assume_aligned(_B, 16_ic);

    /*
     * optimization strategy: communicate divisible problem dimensions. This
     * gives the compiler static facts about index arithmetic and avoids extra
     * boundary handling in this optimized divisible-size path.
     */
    auto M = ct::assume_divisible(_M, 16_ic);
    auto N = ct::assume_divisible(_N, 16_ic);
    auto K = ct::assume_divisible(_K, 16_ic);

    auto a_span = ct::tensor_span{A, ct::extents{M, K}};
    auto b_span = ct::tensor_span{B, ct::extents{K, N}};
    auto c_span = ct::tensor_span{C, ct::extents{M, N}};

    auto a_view = ct::partition_view{a_span, ct::shape<TILE_BLOCK_M, TILE_BLOCK_K>{}};
    auto b_view = ct::partition_view{b_span, ct::shape<TILE_BLOCK_K, TILE_BLOCK_N>{}};
    auto c_view = ct::partition_view{c_span, ct::shape<TILE_BLOCK_M, TILE_BLOCK_N>{}};

    auto [pid_m, pid_n, dummy] = ct::bid();

    auto acc = ct::zeros<ct::tile<float, ct::shape<TILE_BLOCK_M, TILE_BLOCK_N>>>();
    
    /*
     * optimization strategy: use ct::irange for the K traversal. This gives the
     * compiler a Tile-aware structured loop form for iterating through the
     * reduction dimension, unlike the plain C++ loop used in matmul_naive.
     */
    int num_k_blocks = (K + TILE_BLOCK_K - 1) / TILE_BLOCK_K;
    for (auto k_block : ct::irange(0, num_k_blocks)) {
        ct::tile<__half, ct::shape<TILE_BLOCK_M, TILE_BLOCK_K>> a_tile;
        ct::tile<__half, ct::shape<TILE_BLOCK_K, TILE_BLOCK_N>> b_tile;
        
        /*
         * optimization strategy: attach a latency hint to the A tile load.
         * Tile optimization hints can appertain to expression statements and
         * influence scheduling decisions for that construct. The non-masked
         * load also lets the compiler assume every element in this tile is
         * in bounds; the host only launches this kernel for divisible sizes.
         */
        [[
            cutile::hint(0, latency=LOAD_LATENCY),
        ]]
        a_tile = a_view.load(pid_m, k_block);

        /*
         * optimization strategy: attach the same latency hint to the B tile
         * load so the compiler can schedule both input streams with the
         * expected memory latency in mind. As above, this relies on the
         * optimized kernel's divisibility precondition.
         */
        [[
            cutile::hint(0, latency=LOAD_LATENCY),
        ]] 
        b_tile = b_view.load(k_block, pid_n);

        acc = ct::mma(a_tile, b_tile, acc);
    }
    
    /*
     * optimization strategy: attach a store latency hint to the final C tile
     * store. This mirrors the load hints and gives the compiler information
     * about the expected memory latency of the output path. The non-masked
     * store also avoids boundary masking under the same divisibility
     * precondition.
     */
    [[
        cutile::hint(0, latency=STORE_LATENCY),
    ]]
    c_view.store(acc, pid_m, pid_n);
}

void run_with_size(int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0) {
    std::fprintf(stderr,
                 "M, N, and K must be positive (got M=%d, N=%d, K=%d).\n",
                 M, N, K);
    std::exit(EXIT_FAILURE);
  }

  // allocate and initialize FP16 inputs on the host
  std::vector<__half> h_A(M * K), h_B(K * N);
  
  srand(42);
  for (int i = 0; i < M * K; i++) h_A[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
  for (int i = 0; i < K * N; i++) h_B[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
  std::vector<float> h_expected;
  if (use_validation()) {
    // build the expected result once, then reuse it for both device kernels
    h_expected.resize(M * N);
    matmul_cpu(h_expected.data(), h_A.data(), h_B.data(), M, N, K);
  }

  __half *d_A, *d_B;
  float *d_C;
  checkCudaErrors(cudaMalloc(&d_A, M * K * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_B, K * N * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_C, M * N * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));

  bool passed = true;
  // both kernels run only when the matmul()'s precondition is met
  if (M % TILE_BLOCK_M == 0 && N % TILE_BLOCK_N == 0 && K % TILE_BLOCK_K == 0) {
    dim3 grid(M / TILE_BLOCK_M, N / TILE_BLOCK_N);
    auto run_kernel = [&](const char* name, auto kernel_launch) {
      // clear C before each benchmarked kernel for independent validation
      checkCudaErrors(cudaMemset(d_C, 0, M * N * sizeof(float)));

      BenchmarkResult result = run_benchmark(name,
          [&]() {
            kernel_launch();
            checkCudaErrors(cudaGetLastError());
          },
          [&]() {
            std::vector<float> h_C(M * N);
            checkCudaErrors(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float),
                                       cudaMemcpyDeviceToHost));

            return verify_matmul_result(name, h_C.data(), h_expected.data(), M, N);
          },
          M, N, K);

      print_result(result);
      return result.correct;
    };

    // run and validate the optimized kernel
    passed &= run_kernel("matmul", [&]() {
      matmul<<<grid, 1>>>(d_C, d_A, d_B, M, N, K);
    });

    // run and validate the baseline kernel
    passed &= run_kernel("matmul_naive", [&]() {
      matmul_naive<<<grid, 1>>>(d_C, d_A, d_B, M, N, K);
    });

  } else {
    std::fprintf(stderr,
                 "Skipping M=%d, N=%d, K=%d as the optimized kernel assumes "
                 "dimensions divisible by TILE_BLOCK_M=%d, TILE_BLOCK_N=%d, "
                 "TILE_BLOCK_K=%d.\n",
                 M, N, K, TILE_BLOCK_M, TILE_BLOCK_N, TILE_BLOCK_K);
    passed = false;
  }

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));

  if (!passed) {
    std::exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  parse_benchmark_args(argc, argv);
  run_with_size(1024, 1024, 1024);
}
