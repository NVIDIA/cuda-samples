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
 * CUDA Tile C++ matrix multiplication kernel used by tileMatmulAutotuner.
 *
 * This sample implements a tiled FP16 -> FP32 matrix multiplication with
 * ct::partition_view and ct::mma. The autotuner compiles this file repeatedly
 * with TILE_BLOCK_M, TILE_BLOCK_N, TILE_BLOCK_K, LOAD_LATENCY, and
 * STORE_LATENCY defined on the compiler command line.
 *
 * Approach:
 *   - Uses ct::tensor_span and ct::partition_view for blocked access.
 *   - Uses a K-dimension accumulation loop with ct::mma.
 *   - Loads FP16 inputs into tiles and accumulates into FP32.
 */

#include "cuda_tile.h"
#include <cuda_fp16.h>

namespace ct = cuda::tiles;

extern "C" __tile_global__ void matmul_tile(float* __restrict__ _C,
                                             const __half* __restrict__ _A,
                                             const __half* __restrict__ _B,
                                             int _M, int _N, int _K) {
    float* C = ct::assume_aligned<16>(_C);
    const __half* A = ct::assume_aligned<16>(_A);
    const __half* B = ct::assume_aligned<16>(_B);
    auto M = ct::assume_divisible<16>(_M);
    auto N = ct::assume_divisible<16>(_N);
    auto K = ct::assume_divisible<16>(_K);
    
    // Create tensor spans with runtime shapes (FP16 for A and B)
    auto a_span = ct::tensor_span{A, ct::extents{M, K}};
    auto b_span = ct::tensor_span{B, ct::extents{K, N}};
    auto c_span = ct::tensor_span{C, ct::extents{M, N}};
    
    // Create partition views with compile-time block sizes
    auto a_view = ct::partition_view{a_span, ct::shape<TILE_BLOCK_M, TILE_BLOCK_K>{}};
    auto b_view = ct::partition_view{b_span, ct::shape<TILE_BLOCK_K, TILE_BLOCK_N>{}};
    auto c_view = ct::partition_view{c_span, ct::shape<TILE_BLOCK_M, TILE_BLOCK_N>{}};
    
    // get block indices from the 2D grid
    auto [pid_m, pid_n, dummy] = ct::bid();
    
    // initialize FP32 accumulator
    auto acc = ct::zeros<ct::tile<float, ct::shape<TILE_BLOCK_M, TILE_BLOCK_N>>>();
    
    // loop over the K dimension in blocks
    int num_k_blocks = (K + TILE_BLOCK_K - 1) / TILE_BLOCK_K;
    for (auto k_block : ct::irange(0, num_k_blocks)) {
        ct::tile<__half, ct::shape<TILE_BLOCK_M, TILE_BLOCK_K>> a_tile;
        ct::tile<__half, ct::shape<TILE_BLOCK_K, TILE_BLOCK_N>> b_tile;
        
        // load blocks of A and B (FP16)
        [[
            cutile::hint(0, latency=LOAD_LATENCY),
        ]]
        a_tile = a_view.load(pid_m, k_block);

        [[
            cutile::hint(0, latency=LOAD_LATENCY),
        ]] 
        b_tile = b_view.load(k_block, pid_n);
        
        // accumulate: acc += A_block @ B_block (FP16 inputs, FP32 accumulator)
        // ct::mma handles mixed precision: FP16 operands with FP32 accumulator.
        acc = ct::mma(a_tile, b_tile, acc);
    }
    
    // store result (FP32)
    [[
        cutile::hint(0, latency=STORE_LATENCY),
    ]]
    c_view.store(acc, pid_m, pid_n);
}
