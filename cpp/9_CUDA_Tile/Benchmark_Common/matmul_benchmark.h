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
 * This file provides matrix multiplication benchmark helpers shared by
 * tileMatmul and tileMatmulAutotuner.
 */

#ifndef CUDA_TILE_MATMUL_BENCHMARK_H
#define CUDA_TILE_MATMUL_BENCHMARK_H

#include "benchmark.h"

#include <cuda_fp16.h>
#include <cmath>

inline void fill_matmul_metrics(BenchmarkResult& result, int M, int N, int K) {
    // FLOPs: 2 * M * N * K (multiply + add for each output element).
    double flops = 2.0 * M * N * K;
    result.gflops = flops / (result.time_ms * 1e6);

    // Bandwidth: read A, read B, and write C.
    size_t bytes = ((size_t)M * K + (size_t)K * N) * sizeof(__half) +
                   (size_t)M * N * sizeof(float);
    result.bandwidth_gb_s = (bytes / 1e9) / (result.time_ms / 1000.0);
}

// CPU reference implementation (FP16 -> FP32).
inline void matmul_cpu(float* C, const __half* A, const __half* B, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

inline bool verify_matmul_result(const char* name,
                                 const float* h_result,
                                 const float* h_expected,
                                 int M, int N) {
    for (int i = 0; i < M * N; i++) {
        float abs_err = std::abs(h_result[i] - h_expected[i]);
        float rel_err = abs_err / (std::abs(h_expected[i]) + 1e-6f);
        // FP16 has less precision, so allow a larger tolerance.
        if (abs_err > 1e-2f && rel_err > 0.1f) {
            printf("%s verification failed at %d: got %f, expected %f\n",
                   name, i, h_result[i], h_expected[i]);
            return false;
        }
    }

    return true;
}

template<typename KernelFunc, typename ValidateFunc>
inline BenchmarkResult run_benchmark(const char* name,
                                     KernelFunc kernel_launch,
                                     ValidateFunc validate_result,
                                     int M, int N, int K) {
    BenchmarkResult result;
    result.name = name;
    result.time_ms = time_kernel(kernel_launch);
    fill_matmul_metrics(result, M, N, K);
    result.correct = !use_validation() || validate_result();
    return result;
}

#endif // CUDA_TILE_MATMUL_BENCHMARK_H
