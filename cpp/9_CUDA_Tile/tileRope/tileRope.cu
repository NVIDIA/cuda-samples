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
 * This sample demonstrates a Rotary Position Embedding (RoPE) forward pass
 * using CUDA Tile C++. RoPE injects positional information into the query
 * and key projections of an attention layer by rotating pairs of features
 * in the head dimension by per-position angles. This implementation uses
 * the split-half convention: for each token at position 's' the pair
 * (q[i], q[i + D/2]) is rotated by theta = s * 10000^(-2i / D), yielding
 *     q[i]'       = q[i] * cos(theta) - q[i + D/2] * sin(theta)
 *     q[i + D/2]' = q[i] * sin(theta) + q[i + D/2] * cos(theta)
 * Each block handles one (batch, position) token and processes all heads
 * in parallel using 2D tiles over (heads, half_rope_dim). The kernel
 * writes back to q and k in place. A SIMT kernel is used to initialize
 * the inputs and the cos/sin tables.
 */

#include "helper_cuda.h"

#include "cuda_tile.h"
#include "cuda_fp16.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

/* Compile-time sample shape: one block per (batch, position) token. */
constexpr int BATCH         = 1;
constexpr int Q_HEADS       = 8;
constexpr int K_HEADS       = 8;
constexpr int SEQ_LEN       = 64;
constexpr int HEAD_DIM      = 64;
constexpr int HALF_ROPE_DIM = HEAD_DIM / 2;
constexpr int BLOCK_QH      = Q_HEADS;
constexpr int BLOCK_KH      = K_HEADS;
constexpr int BLOCK_HD      = HALF_ROPE_DIM;
constexpr int COS_BS        = 1;

constexpr std::size_t Q_SIZE   = (std::size_t)BATCH * Q_HEADS * SEQ_LEN * HEAD_DIM;
constexpr std::size_t K_SIZE   = (std::size_t)BATCH * K_HEADS * SEQ_LEN * HEAD_DIM;
constexpr std::size_t COS_SIZE = (std::size_t)COS_BS * SEQ_LEN * HALF_ROPE_DIM;

constexpr std::size_t cmax(std::size_t a, std::size_t b) { return a > b ? a : b; }
constexpr std::size_t INIT_N = cmax(Q_SIZE, cmax(K_SIZE, COS_SIZE));

/* Initializes q, k and the cos/sin tables on device. The cos/sin tables
 * are laid out as (COS_BS, SEQ_LEN, HALF_ROPE_DIM): one entry per
 * (batch, position, frequency-index) triple. */
__global__ void initializeInputs(__half* q, __half* k, __half* cos, __half* sin) {
  std::size_t tid = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < Q_SIZE) {
    int d = (int)(tid % HEAD_DIM);
    q[tid] = __half{float(d % 11) / 10.0f - 0.5f};
  }
  if (tid < K_SIZE) {
    int d = (int)(tid % HEAD_DIM);
    k[tid] = __half{float(d % 13) / 10.0f - 0.5f};
  }
  if (tid < COS_SIZE) {
    int i = (int)(tid % HALF_ROPE_DIM);
    int s = (int)((tid / HALF_ROPE_DIM) % SEQ_LEN);
    float exponent = -2.0f * float(i) / float(HEAD_DIM);
    float theta = float(s) * powf(10000.0f, exponent);
    cos[tid] = __half{cosf(theta)};
    sin[tid] = __half{sinf(theta)};
  }
}

/* RoPE forward kernel - processes all heads at once using 2D tiles. One
 * block handles one (batch, seq) position. Tile 0 along the head_dim axis
 * spans [0:BLOCK_HD) (the first half) and tile 1 spans [BLOCK_HD:2*BLOCK_HD)
 * (the second half), so the rotation pairs are (q[i], q[i + D/2]). */
template <typename T, int BATCH_, int Q_HEADS_, int K_HEADS_,
          int BLOCK_QH_, int BLOCK_KH_, int BLOCK_HD_,
          int HALF_ROPE_DIM_, int HEAD_DIM_, int COS_BS_, int SEQ_LEN_>
__tile_global__ void rope(T* __restrict__ q,
                          T* __restrict__ k,
                          T* __restrict__ cos,
                          T* __restrict__ sin) {
  namespace ct = cuda::tiles;

  q   = ct::assume_aligned<16>(q);
  k   = ct::assume_aligned<16>(k);
  cos = ct::assume_aligned<16>(cos);
  sin = ct::assume_aligned<16>(sin);

  int pid           = ct::bid().x;
  int batch_idx     = pid / SEQ_LEN_;
  int row_idx       = pid % SEQ_LEN_;
  int cos_batch_idx = (COS_BS_ == 1) ? 0 : batch_idx;

  auto pCos = ct::partition_view(ct::tensor_span{cos, ct::extents{COS_BS_, SEQ_LEN_, HALF_ROPE_DIM_}},
                                 ct::shape<1, 1, BLOCK_HD_>{});
  auto cos_loaded = pCos.load(cos_batch_idx, row_idx, 0);
  auto cos_row    = ct::reshape<ct::shape<1, BLOCK_HD_>>(cos_loaded);

  auto pSin = ct::partition_view(ct::tensor_span{sin, ct::extents{COS_BS_, SEQ_LEN_, HALF_ROPE_DIM_}},
                                 ct::shape<1, 1, BLOCK_HD_>{});
  auto sin_loaded = pSin.load(cos_batch_idx, row_idx, 0);
  auto sin_row    = ct::reshape<ct::shape<1, BLOCK_HD_>>(sin_loaded);

  /* Process Q. Tile indices 0 and 1 along the head_dim axis cover
   * [0:2*BLOCK_HD) == [0:rope_dim); elements past rope_dim are unchanged. */
  auto pQ = ct::partition_view(ct::tensor_span{q, ct::extents{BATCH_, Q_HEADS_, SEQ_LEN_, HEAD_DIM_}},
                               ct::shape<1, BLOCK_QH_, 1, BLOCK_HD_>{});
  auto q_tile_1_loaded = pQ.load(batch_idx, 0, row_idx, 0);
  auto q_tile_1        = ct::reshape<ct::shape<BLOCK_QH_, BLOCK_HD_>>(q_tile_1_loaded);

  auto q_tile_2_loaded = pQ.load(batch_idx, 0, row_idx, 1);
  auto q_tile_2        = ct::reshape<ct::shape<BLOCK_QH_, BLOCK_HD_>>(q_tile_2_loaded);

  auto cos_bcast_q = ct::broadcast<ct::shape<BLOCK_QH_, BLOCK_HD_>>(cos_row);
  auto sin_bcast_q = ct::broadcast<ct::shape<BLOCK_QH_, BLOCK_HD_>>(sin_row);

  /* y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin] */
  auto new_q_tile_1 = q_tile_1 * cos_bcast_q - q_tile_2 * sin_bcast_q;
  auto new_q_tile_2 = q_tile_2 * cos_bcast_q + q_tile_1 * sin_bcast_q;

  auto new_q_tile_1_reshaped = ct::reshape<ct::shape<1, BLOCK_QH_, 1, BLOCK_HD_>>(new_q_tile_1);
  auto new_q_tile_2_reshaped = ct::reshape<ct::shape<1, BLOCK_QH_, 1, BLOCK_HD_>>(new_q_tile_2);

  pQ.store(new_q_tile_1_reshaped, batch_idx, 0, row_idx, 0);
  pQ.store(new_q_tile_2_reshaped, batch_idx, 0, row_idx, 1);

  /* Process K (in place). */
  auto pK = ct::partition_view(ct::tensor_span{k, ct::extents{BATCH_, K_HEADS_, SEQ_LEN_, HEAD_DIM_}},
                               ct::shape<1, BLOCK_KH_, 1, BLOCK_HD_>{});
  auto k_tile_1_loaded = pK.load(batch_idx, 0, row_idx, 0);
  auto k_tile_1        = ct::reshape<ct::shape<BLOCK_KH_, BLOCK_HD_>>(k_tile_1_loaded);

  auto k_tile_2_loaded = pK.load(batch_idx, 0, row_idx, 1);
  auto k_tile_2        = ct::reshape<ct::shape<BLOCK_KH_, BLOCK_HD_>>(k_tile_2_loaded);

  auto cos_bcast_k = ct::broadcast<ct::shape<BLOCK_KH_, BLOCK_HD_>>(cos_row);
  auto sin_bcast_k = ct::broadcast<ct::shape<BLOCK_KH_, BLOCK_HD_>>(sin_row);

  auto new_k_tile_1 = k_tile_1 * cos_bcast_k - k_tile_2 * sin_bcast_k;
  auto new_k_tile_2 = k_tile_2 * cos_bcast_k + k_tile_1 * sin_bcast_k;

  auto new_k_tile_1_reshaped = ct::reshape<ct::shape<1, BLOCK_KH_, 1, BLOCK_HD_>>(new_k_tile_1);
  auto new_k_tile_2_reshaped = ct::reshape<ct::shape<1, BLOCK_KH_, 1, BLOCK_HD_>>(new_k_tile_2);

  pK.store(new_k_tile_1_reshaped, batch_idx, 0, row_idx, 0);
  pK.store(new_k_tile_2_reshaped, batch_idx, 0, row_idx, 1);
}

/* CPU reference matching the split-half convention used by the kernel:
 * (q[i], q[i + D/2]) is rotated by the angle stored at cos/sin[cb, s, i]. */
static bool verify(const __half* h_in,
                   const __half* h_out,
                   const __half* h_cos,
                   const __half* h_sin,
                   int heads,
                   const char* name) {
  for (int b = 0; b < BATCH; ++b) {
    for (int h = 0; h < heads; ++h) {
      for (int s = 0; s < SEQ_LEN; ++s) {
        for (int i = 0; i < HALF_ROPE_DIM; ++i) {
          std::size_t base = (((std::size_t)b * heads + h) * SEQ_LEN + s) * HEAD_DIM;
          std::size_t i1   = base + i;
          std::size_t i2   = base + i + HALF_ROPE_DIM;
          int cb           = (COS_BS == 1) ? 0 : b;
          std::size_t ci   = ((std::size_t)cb * SEQ_LEN + s) * HALF_ROPE_DIM + i;

          double q1 = (double)(float)h_in[i1];
          double q2 = (double)(float)h_in[i2];
          double c  = (double)(float)h_cos[ci];
          double si = (double)(float)h_sin[ci];

          __half exp1 = __half{(float)(q1 * c - q2 * si)};
          __half exp2 = __half{(float)(q2 * c + q1 * si)};

          float diff1 = std::fabs((float)h_out[i1] - (float)exp1);
          float diff2 = std::fabs((float)h_out[i2] - (float)exp2);

          if (diff1 > 1e-1f || diff2 > 1e-1f) {
            printf("Mismatch in %s at (b=%d, h=%d, s=%d, i=%d):\n", name, b, h, s, i);
            printf("  Expected: %s[%d,%d,%d,%d]=%f, %s[%d,%d,%d,%d]=%f\n",
                   name, b, h, s, i,                  (float)exp1,
                   name, b, h, s, i + HALF_ROPE_DIM,  (float)exp2);
            printf("  Actual:   %s[%d,%d,%d,%d]=%f, %s[%d,%d,%d,%d]=%f\n",
                   name, b, h, s, i,                  (float)h_out[i1],
                   name, b, h, s, i + HALF_ROPE_DIM,  (float)h_out[i2]);
            return false;
          }
        }
      }
    }
  }
  return true;
}

int main() {
  __half* d_q   = nullptr;
  __half* d_k   = nullptr;
  __half* d_cos = nullptr;
  __half* d_sin = nullptr;

  checkCudaErrors(cudaMalloc(&d_q,   Q_SIZE   * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_k,   K_SIZE   * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_cos, COS_SIZE * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_sin, COS_SIZE * sizeof(__half)));

  int threads_per_block = 256;
  int num_blocks        = (int)((INIT_N + threads_per_block - 1) / threads_per_block);

  initializeInputs<<<num_blocks, threads_per_block>>>(d_q, d_k, d_cos, d_sin);
  checkCudaErrors(cudaGetLastError());

  /* Snapshot the inputs before the in-place kernel mutates them. */
  __half* h_q_in = new __half[Q_SIZE];
  __half* h_k_in = new __half[K_SIZE];
  __half* h_cos  = new __half[COS_SIZE];
  __half* h_sin  = new __half[COS_SIZE];
  checkCudaErrors(cudaMemcpy(h_q_in, d_q,   Q_SIZE   * sizeof(__half), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_k_in, d_k,   K_SIZE   * sizeof(__half), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_cos,  d_cos, COS_SIZE * sizeof(__half), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_sin,  d_sin, COS_SIZE * sizeof(__half), cudaMemcpyDeviceToHost));

  rope<__half, BATCH, Q_HEADS, K_HEADS, BLOCK_QH, BLOCK_KH, BLOCK_HD,
       HALF_ROPE_DIM, HEAD_DIM, COS_BS, SEQ_LEN>
      <<<BATCH * SEQ_LEN>>>(d_q, d_k, d_cos, d_sin);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaDeviceSynchronize());

  __half* h_q_out = new __half[Q_SIZE];
  __half* h_k_out = new __half[K_SIZE];
  checkCudaErrors(cudaMemcpy(h_q_out, d_q, Q_SIZE * sizeof(__half), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_k_out, d_k, K_SIZE * sizeof(__half), cudaMemcpyDeviceToHost));

  if (!verify(h_q_in, h_q_out, h_cos, h_sin, Q_HEADS, "Q")) return 1;
  if (!verify(h_k_in, h_k_out, h_cos, h_sin, K_HEADS, "K")) return 1;

  printf("Success! RoPE matches expected results.\n");

  checkCudaErrors(cudaFree(d_q));
  checkCudaErrors(cudaFree(d_k));
  checkCudaErrors(cudaFree(d_cos));
  checkCudaErrors(cudaFree(d_sin));

  delete[] h_q_in;
  delete[] h_k_in;
  delete[] h_cos;
  delete[] h_sin;
  delete[] h_q_out;
  delete[] h_k_out;
}
