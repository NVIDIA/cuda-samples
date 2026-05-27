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
 * This sample demonstrates a persistent LayerNorm forward pass using
 * CUDA Tile C++:  y = (x - mean) * rsqrt(var + eps) * weight + bias.
 * The grid launches NUM_SMS persistent blocks; each block walks the
 * row dimension with a grid-stride loop, processing BLOCK_N rows x
 * BLOCK_D cols per iteration and striding by NUM_SMS * BLOCK_N rows.
 * Per-row mean and rstd are reduced over the column dimension and
 * (when COMPUTE_MEAN_AND_RSTD and TRAINING are enabled) saved to
 * float32 side buffers. N, D, NUM_SMS, and EPS are template NTTPs so
 * the tile compiler can fold the loop step, the (1/D) reciprocal,
 * partition_view extents, and the (var + eps) broadcast.  A SIMT
 * kernel is used to initialize X, W, and B on device.
 */

#include "helper_cuda.h"
#include "cuda_tile.h"
#include "cuda_fp16.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

/* SIMT initializer for X (N x D), W (D,), B (D,) with deterministic data. */
__global__ void initializeInputs(__half* X, __half* W, __half* B,
                                 int N, int D) {
  auto idx   = blockIdx.x * blockDim.x + threadIdx.x;
  auto total = N * D;
  if (idx < total) {
    int m = idx / D;
    int n = idx - m * D;
    X[idx] = __half{float((m + n) % 7) - 3.5f};
  }
  if (idx < D) {
    W[idx] = __half{1.0f + 0.1f * float(idx % 5)};
    B[idx] = __half{0.1f * float(idx % 3)};
  }
}

template<typename T,
         int BLOCK_N,
         int BLOCK_D,
         bool TRAINING,
         bool COMPUTE_MEAN_AND_RSTD,
         int N,
         int D,        // compile-time so `(... / D)` and partition_view extents fold.
         int NUM_SMS,  // compile-time so the persistent for-loop step is constant.
         float EPS>    // compile-time so the (var + eps) reshape/broadcast hoists out of the for-loop
[[ using cutile : hint(0, num_cta_in_cga=1) ]]
__tile_global__ void persistent_layer_norm_fwd_kernel(
    const T* __restrict__ X,       // (N, D)
    T* __restrict__ Y,             // (N, D)
    const T* __restrict__ W,       // (D,)
    const T* __restrict__ B,       // (D,)
    float* __restrict__ Mean,      // (N,)
    float* __restrict__ Rstd       // (N,)
) {
    namespace ct = cuda::tiles;

    using f32_N     = ct::tile<float, ct::shape<BLOCK_N>>;

    X    = ct::assume_aligned<16>(X);
    Y    = ct::assume_aligned<16>(Y);
    W    = ct::assume_aligned<16>(W);
    B    = ct::assume_aligned<16>(B);

    int pid = ct::bid().x;
    constexpr int upper_bound = (N + BLOCK_N - 1) / BLOCK_N;

    // Partitioned views with compile-time extents (N, D are template NTTPs).
    using ExtND = ct::extents<uint32_t, static_cast<uint32_t>(N), static_cast<uint32_t>(D)>;
    using ExtD  = ct::extents<uint32_t, static_cast<uint32_t>(D)>;
    using ExtN  = ct::extents<uint32_t, static_cast<uint32_t>(N)>;
    auto pX = ct::partition_view(
        ct::tensor_span{X, ExtND{}},
        ct::shape<BLOCK_N, BLOCK_D>{});
    auto pY = ct::partition_view(
        ct::tensor_span{Y, ExtND{}},
        ct::shape<BLOCK_N, BLOCK_D>{});
    auto pW = ct::partition_view(
        ct::tensor_span{W, ExtD{}},
        ct::shape<BLOCK_D>{});
    auto pB = ct::partition_view(
        ct::tensor_span{B, ExtD{}},
        ct::shape<BLOCK_D>{});
    auto pMean = ct::partition_view(
        ct::tensor_span{Mean, ExtN{}},
        ct::shape<BLOCK_N>{});
    auto pRstd = ct::partition_view(
        ct::tensor_span{Rstd, ExtN{}},
        ct::shape<BLOCK_N>{});

    // Load weights once (hoisted out of the grid-stride loop).
    auto w = ct::element_cast<float>(pW.load(0));  // (BLOCK_D,)
    auto b = ct::element_cast<float>(pB.load(0));  // (BLOCK_D,)

    // Broadcast weights into (BLOCK_N, BLOCK_D) by reshape to (1, BLOCK_D).
    auto w_bcast = ct::reshape<ct::shape<1, BLOCK_D>>(w);
    auto b_bcast = ct::reshape<ct::shape<1, BLOCK_D>>(b);

    constexpr float inv_D_scalar = 1.0f / static_cast<float>(D);
    auto inv_D_tile = ct::full<f32_N>(inv_D_scalar);
    auto eps_tile   = ct::full<f32_N>(EPS);

    using TileXNxD = ct::tile<T, ct::shape<BLOCK_N, BLOCK_D>>;

    for (auto current_pid : ct::irange(pid, upper_bound, NUM_SMS)) {
        TileXNxD x_tile;
        [[ using cutile : hint(0, latency=4) ]]
        x_tile = pX.load_masked(current_pid, 0);
        auto x = ct::element_cast<float>(x_tile);

        f32_N mean;
        f32_N rstd;

        if constexpr (COMPUTE_MEAN_AND_RSTD) {
            // Step 1: Compute x^2 then sum/mean.  Use the loop-invariant
            // `inv_D_tile` and `eps_tile` (built outside the loop) so the
            // reshape + broadcast of those scalars stays hoisted.
            auto x_squared = x * x;
            auto avg_square_2d = ct::sum<1>(x_squared);
            auto avg_square = ct::reshape<ct::shape<BLOCK_N>>(avg_square_2d) * inv_D_tile;
            auto mean_2d = ct::sum<1>(x);
            mean = ct::reshape<ct::shape<BLOCK_N>>(mean_2d) * inv_D_tile;
            auto var = avg_square - mean * mean;

            rstd = ct::rsqrt(var + eps_tile);

            if constexpr (TRAINING) {
                [[ using cutile : hint(0, allow_tma=false) ]]
                pMean.store_masked(mean, current_pid);
                [[ using cutile : hint(0, allow_tma=false) ]]
                pRstd.store_masked(rstd, current_pid);
            }
        } else {
            mean = pMean.load_masked(current_pid);
            rstd = pRstd.load_masked(current_pid);
        }

        // Broadcast mean/rstd to (BLOCK_N, 1) then rely on implicit broadcast
        // against (BLOCK_N, BLOCK_D).
        auto mean_col = ct::reshape<ct::shape<BLOCK_N, 1>>(mean);
        auto rstd_col = ct::reshape<ct::shape<BLOCK_N, 1>>(rstd);

        auto x_hat = (x - mean_col) * rstd_col;
        auto y_f32 = x_hat * w_bcast + b_bcast;

        auto y_T = ct::element_cast<T>(y_f32);
        [[ using cutile : hint(0, allow_tma=false) ]]
        pY.store_masked(y_T, current_pid, 0);
    }
}

int main() {
  /* BLOCK_D == D so each persistent-loop iteration covers a full row's
   * columns in one tile per row. NUM_SMS is a template NTTP, hence
   * compile-time; 132 matches B200 / Hopper-class GPUs - adjust to
   * match the target device's `multiProcessorCount` for best perf. */
  constexpr int   N = 1024, D = 256;
  constexpr int   BLOCK_N = 4, BLOCK_D = 256;
  constexpr int   NUM_SMS = 132;
  constexpr float EPS     = 1e-5f;

  __half *d_X = nullptr, *d_Y = nullptr, *d_W = nullptr, *d_B = nullptr;
  float  *d_Mean = nullptr, *d_Rstd = nullptr;
  checkCudaErrors(cudaMalloc(&d_X,    N * D * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_Y,    N * D * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_W,    D     * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_B,    D     * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_Mean, N     * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_Rstd, N     * sizeof(float)));

  int init_threads = 256, init_blocks = 1 + ((N * D - 1) / init_threads);
  initializeInputs<<<init_blocks, init_threads>>>(d_X, d_W, d_B, N, D);
  checkCudaErrors(cudaGetLastError());

  /* NUM_SMS is a compile-time NTTP that doubles as the persistent-loop
   * stride; the launch grid x must equal NUM_SMS for correctness.
   * Adjust the constant (and recompile) for one block per SM on a
   * device with a different SM count. */
  persistent_layer_norm_fwd_kernel<__half, BLOCK_N, BLOCK_D,
                                   /*TRAINING=*/true, /*COMPUTE_MEAN_AND_RSTD=*/true,
                                   N, D, NUM_SMS, EPS>
      <<<dim3(NUM_SMS, 1, 1)>>>(d_X, d_Y, d_W, d_B, d_Mean, d_Rstd);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  __half* h_Y        = new __half[N * D];
  __half* h_Y_ref    = new __half[N * D];
  float*  h_Mean     = new float[N];
  float*  h_Rstd     = new float[N];
  float*  h_Mean_ref = new float[N];
  float*  h_Rstd_ref = new float[N];
  checkCudaErrors(cudaMemcpy(h_Y,    d_Y,    N * D * sizeof(__half), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_Mean, d_Mean, N     * sizeof(float),  cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_Rstd, d_Rstd, N     * sizeof(float),  cudaMemcpyDeviceToHost));

  /* CPU reference in double precision; compare with 1e-1 fp16 tolerance
   * for Y and 1e-3 for the float32 Mean/Rstd outputs. */
  for (int m = 0; m < N; ++m) {
    double sum = 0.0, sumsq = 0.0;
    for (int n = 0; n < D; ++n) {
      double x = double(float((m + n) % 7) - 3.5f);
      sum += x; sumsq += x * x;
    }
    double mu      = sum / double(D);
    double var     = sumsq / double(D) - mu * mu;
    double inv_std = 1.0 / std::sqrt(var + double(EPS));
    h_Mean_ref[m]  = float(mu);
    h_Rstd_ref[m]  = float(inv_std);
    for (int n = 0; n < D; ++n) {
      double x = double(float((m + n) % 7) - 3.5f);
      double w = double(1.0f + 0.1f * float(n % 5));
      double b = double(0.1f * float(n % 3));
      h_Y_ref[m * D + n] = __half(float((x - mu) * inv_std * w + b));
    }
  }

  for (int idx = 0; idx < N * D; ++idx) {
    float expected = float(h_Y_ref[idx]), actual = float(h_Y[idx]);
    if (std::fabs(expected - actual) > 1e-1f) {
      printf("Mismatch h_Y[%d]: expected %f, actual %f\n", idx, expected, actual);
      return 1;
    }
  }
  for (int m = 0; m < N; ++m) {
    if (std::fabs(h_Mean_ref[m] - h_Mean[m]) > 1e-3f) {
      printf("Mismatch h_Mean[%d]: expected %f, actual %f\n", m, h_Mean_ref[m], h_Mean[m]);
      return 1;
    }
    if (std::fabs(h_Rstd_ref[m] - h_Rstd[m]) > 1e-3f) {
      printf("Mismatch h_Rstd[%d]: expected %f, actual %f\n", m, h_Rstd_ref[m], h_Rstd[m]);
      return 1;
    }
  }

  printf("Success! Persistent LayerNorm matches expected results.\n");

  checkCudaErrors(cudaFree(d_X));   checkCudaErrors(cudaFree(d_Y));
  checkCudaErrors(cudaFree(d_W));   checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_Mean)); checkCudaErrors(cudaFree(d_Rstd));
  delete[] h_Y;        delete[] h_Y_ref;
  delete[] h_Mean;     delete[] h_Rstd;
  delete[] h_Mean_ref; delete[] h_Rstd_ref;
}
