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
 * This file provides common benchmark utilities for CUDA Tile C++
 * microbenchmarks.
 */

#ifndef CUDA_TILE_BENCHMARK_H
#define CUDA_TILE_BENCHMARK_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// benchmark configuration
// global settings for benchmark behavior, controllable via command line
struct BenchmarkConfig {
    bool use_validation = false;   // --validate enables CPU cross-validation
    int warmup_iters = 5;          // --warmup=N (0 to disable)
    int bench_iters = 20;          // --iters=N, -i N
};

inline BenchmarkConfig& bench_config() {
    static BenchmarkConfig config;
    return config;
}

// convenience accessors
inline bool use_validation() { return bench_config().use_validation; }
inline int warmup_iters() { return bench_config().warmup_iters; }
inline int bench_iters() { return bench_config().bench_iters; }

// Parse command line options. Call from main() before benchmarks.
inline void parse_benchmark_args(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--validate") == 0) {
            bench_config().use_validation = true;
        } else if (strcmp(argv[i], "--skip-warmup") == 0) {
            bench_config().warmup_iters = 0;
        } else if (strncmp(argv[i], "--warmup=", 9) == 0) {
            bench_config().warmup_iters = atoi(argv[i] + 9);
        } else if (strncmp(argv[i], "--iters=", 8) == 0) {
            bench_config().bench_iters = atoi(argv[i] + 8);
        } else if (strcmp(argv[i], "-i") == 0) {
            // Parse "-i N" where the next argument is the value.
            if (i + 1 < argc) {
                bench_config().bench_iters = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: -i requires an argument\n");
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Benchmark options:\n");
            printf("  --validate             Enable CPU cross-validation\n");
            printf("  --skip-warmup          Disable warmup iterations\n");
            printf("  --warmup=N             Warmup iterations (default: 5)\n");
            printf("  -i N, --iters=N        Benchmark iterations (default: 20)\n");
            exit(0);
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Error: unknown option '%s'\n", argv[i]);
            fprintf(stderr, "Try '--help' for usage information.\n");
            exit(EXIT_FAILURE);
        }
    }
    
    // print active configuration
    if (!bench_config().use_validation) {
        printf("Note: CPU cross-validation disabled\n");
    }
    if (bench_config().warmup_iters == 0) {
        printf("Note: warmup disabled, iters=%d\n", bench_config().bench_iters);
    } else if (bench_config().warmup_iters != 5 || bench_config().bench_iters != 20) {
        printf("Note: warmup=%d, iters=%d\n", 
               bench_config().warmup_iters, bench_config().bench_iters);
    }
}

// CUDA error checking
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// device information
inline void print_device_info() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    
    int memoryClockRateKHz;
    cudaError_t status = cudaDeviceGetAttribute(&memoryClockRateKHz, cudaDevAttrMemoryClockRate, 0);
    if (status == cudaSuccess) {
        printf("Memory Bandwidth: %.0f GB/s (theoretical peak)\n",
               2.0 * memoryClockRateKHz * (prop.memoryBusWidth / 8) / 1e6);
    }
}

// timing utilities
class CudaTimer {
public:
    CudaTimer() {
        CHECK_CUDA(cudaEventCreate(&start_));
        CHECK_CUDA(cudaEventCreate(&stop_));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        CHECK_CUDA(cudaEventRecord(start_));
    }
    
    void stop() {
        CHECK_CUDA(cudaEventRecord(stop_));
        CHECK_CUDA(cudaEventSynchronize(stop_));
    }
    
    float elapsed_ms() const {
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
    
private:
    cudaEvent_t start_, stop_;
};

// Time a kernel launch, returning average time per iteration in milliseconds.
// Uses global bench_config() for warmup/iteration counts.
template<typename KernelFunc>
inline double time_kernel(KernelFunc kernel_launch) {
    // warmup
    if (warmup_iters() > 0) {
        for (int i = 0; i < warmup_iters(); i++) {
            kernel_launch();
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    // benchmark
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < bench_iters(); i++) {
        kernel_launch();
    }
    timer.stop();
    
    return timer.elapsed_ms() / bench_iters();
}

// benchmark result structure
struct BenchmarkResult {
    const char* name;
    double time_ms;
    double bandwidth_gb_s;
    double gflops;
    bool correct;
    
    BenchmarkResult() : name(nullptr), time_ms(0), bandwidth_gb_s(0), gflops(0), correct(false) {}
};

// result printing
inline void print_result(const BenchmarkResult& r) {
    const char* status = use_validation() ? (r.correct ? "[OK]" : "[FAIL]") : "";
    if (r.gflops > 0) {
        printf("  %-42s: %7.3f ms, %7.1f GB/s, %6.1f GFLOPS %s\n",
               r.name, r.time_ms, r.bandwidth_gb_s, r.gflops, status);
    } else {
        printf("  %-42s: %7.3f ms, %7.1f GB/s %s\n",
               r.name, r.time_ms, r.bandwidth_gb_s, status);
    }
}

#endif // CUDA_TILE_BENCHMARK_H
