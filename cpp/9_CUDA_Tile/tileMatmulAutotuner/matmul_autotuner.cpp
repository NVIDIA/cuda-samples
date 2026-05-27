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
 * CUDA Tile C++ matrix multiplication autotuner.
 *
 * This sample is the host-side driver for the tiled matrix multiplication
 * kernel in matmul.cu. It compiles that kernel repeatedly with different
 * TILE_BLOCK_M, TILE_BLOCK_N, TILE_BLOCK_K, LOAD_LATENCY, and STORE_LATENCY
 * values configured in a search-space file, derives the launch grid from the
 * selected tile size, and reports the fastest configuration for the requested
 * problem size.
 *
 * Backend flow:
 *   - NVRTC compiles matmul.cu to TileIR, then invokes tileiras to produce a
 *     cubin image.
 *   - NVCC compiles matmul.cu as a standalone source file and reuses the
 *     generated Tile cubin artifact.
 *   - Both paths load the resulting cubin with the CUDA Driver API and launch
 *     the same matmul_tile entry point.
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>

#include "backend_common.h"
#include "backend_nvcc.h"
#include "backend_nvrtc.h"
#include "matmul_benchmark.h"
#include <helper_cuda_drvapi.h>

// global SM value (compute capability)
static int smValue = 0;
static constexpr const char *kMatmulKernelName = "matmul_tile";

CompilerBackend parseCompilerBackendValue(const char *value) {
    if (std::strcmp(value, "nvrtc") == 0) {
        return CompilerBackend::NVRTC;
    }
    if (std::strcmp(value, "nvcc") == 0) {
        return CompilerBackend::NVCC;
    }

    fprintf(stderr, "Error: unsupported backend '%s'\n", value);
    fprintf(stderr, "Expected 'nvrtc' or 'nvcc'.\n");
    exit(EXIT_FAILURE);
}

void printCompilerOptions() {
    printf("Backend options:\n");
    printf("  --backend=nvrtc|nvcc   Select backend (default: NVRTC)\n");
    printf("\n");
}

CompilerBackend parseCompilerBackendArgs(int argc, char** argv, std::vector<char*>& benchmark_argv) {
    CompilerBackend compiler_backend = CompilerBackend::NVRTC;
    benchmark_argv.clear();
    benchmark_argv.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            printCompilerOptions();
            benchmark_argv.push_back(argv[i]);
        } else if (std::strncmp(argv[i], "--backend=", 10) == 0) {
            compiler_backend = parseCompilerBackendValue(argv[i] + 10);
        } else if (std::strcmp(argv[i], "--backend") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", argv[i]);
                exit(EXIT_FAILURE);
            }
            compiler_backend = parseCompilerBackendValue(argv[++i]);
        } else {
            benchmark_argv.push_back(argv[i]);
        }
    }

    return compiler_backend;
}

void setSMValue() {
    CUdevice device;
    int major = 0, minor = 0;

    // initialize the CUDA Driver API
    checkCudaErrors(cuInit(0));

    // get the first device (device 0)
    checkCudaErrors(cuDeviceGet(&device, 0));
    checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    printf("GPU Compute Capability: %d.%d\n", major, minor);
    smValue = major * 10 + minor;
}

CompiledKernel compileFile(const char *filename,
                           int block_m, int block_n, int block_k,
                           CompilerBackend compiler_backend,
                           const std::vector<std::string>& extra_flags = {}) {
    if (compiler_backend == CompilerBackend::NVCC) {
        return compileFileWithNVCC(filename, smValue, block_m, block_n, block_k, extra_flags);
    }
    return compileFileWithNVRTC(filename, smValue, block_m, block_n, block_k, extra_flags);
}

void loadAndExecuteKernel(const CompiledKernel& compiled_kernel,
                          CUdeviceptr d_A, CUdeviceptr d_B, CUdeviceptr d_C,
                          int M, int N, int K,
                          unsigned int gridDimX, unsigned int gridDimY, unsigned int sMem) {
    CUmodule module;
    CUfunction kernel_addr;

    void* args[] = {
        (void*)&d_C,
        (void*)&d_A,
        (void*)&d_B,
        (void*)&M,
        (void*)&N,
        (void*)&K
    };

    checkCudaErrors(cuModuleLoadData(&module, compiled_kernel.image.data()));

    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, kMatmulKernelName));
    checkCudaErrors(cuLaunchKernel(kernel_addr,
                gridDimX, gridDimY, 1,  // grid dim
                1, 1, 1,                // block dim
                sMem, 0,                // shared mem, stream
                args,                   // arguments
                NULL));
    checkCudaErrors(cuCtxSynchronize());

    // cleanup
    checkCudaErrors(cuModuleUnload(module));
}

void autotuner(int M, int N, int K,
               const char *kernel_file,
               const SearchSpace& search_space,
               CompilerBackend compiler_backend) {
    printf("\n=== Matrix: C[%dx%d] = A[%dx%d] x B[%dx%d] (FP16->FP32) ===\n",
           M, N, M, K, K, N);
    printf("    FLOPs: %.2f GFLOP\n", 2.0 * M * N * K / 1e9);

    // allocate and initialize (FP16 for A and B)
    std::vector<__half> h_A(M * K), h_B(K * N);
    std::vector<float> h_C;

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half((float)rand() / RAND_MAX - 0.5f);

    // compute the CPU reference unless validation is disabled
    if (use_validation()) {
        h_C.resize(M * N);
        matmul_cpu(h_C.data(), h_A.data(), h_B.data(), M, N, K);
    }

    // allocate device memory
    CUdeviceptr d_A, d_B, d_C;
    checkCudaErrors(cuMemAlloc(&d_A, M * K * sizeof(__half)));
    checkCudaErrors(cuMemAlloc(&d_B, K * N * sizeof(__half)));
    checkCudaErrors(cuMemAlloc(&d_C, M * N * sizeof(float)));
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A.data(), M * K * sizeof(__half)));
    checkCudaErrors(cuMemcpyHtoD(d_B, h_B.data(), K * N * sizeof(__half)));

    // helper lambda to clear output
    auto clear_output = [&]() {
        std::vector<float> zeros(M * N, 0.0f);
        checkCudaErrors(cuMemcpyHtoD(d_C, zeros.data(), M * N * sizeof(float)));
    };

    struct AutotuneResult {
        int block_m;
        int block_n;
        int block_k;
        int grid_x;
        int grid_y;
        int load_latency;
        int store_latency;
        BenchmarkResult result;
    };
    std::vector<AutotuneResult> autotune_results;

    size_t config_count = 0;
    size_t total_configs = search_space.tile_options.size() *
                           search_space.load_latency_options.size() *
                           search_space.store_latency_options.size();

    for (const auto& tile : search_space.tile_options) {
        int grid_x = ceilDiv(M, tile.block_m);
        int grid_y = ceilDiv(N, tile.block_n);
        for (int load_lat : search_space.load_latency_options) {
            for (int store_lat : search_space.store_latency_options) {
                config_count++;
                printf("  [%zu/%zu] ", config_count, total_configs);

                std::vector<std::string> compile_flags = {
                    "-DLOAD_LATENCY=" + std::to_string(load_lat),
                    "-DSTORE_LATENCY=" + std::to_string(store_lat)
                };

                CompiledKernel compiled_kernel = compileFile(kernel_file,
                                                             tile.block_m, tile.block_n, tile.block_k,
                                                             compiler_backend,
                                                             compile_flags);

                clear_output();

                std::string config_name = "bm=" + std::to_string(tile.block_m) +
                                          ",bn=" + std::to_string(tile.block_n) +
                                          ",bk=" + std::to_string(tile.block_k) +
                                          ",gx=" + std::to_string(grid_x) +
                                          ",gy=" + std::to_string(grid_y) +
                                          ",ld=" + std::to_string(load_lat) +
                                          ",st=" + std::to_string(store_lat);
                auto result = run_benchmark(config_name.c_str(),
                    [&]() {
                        loadAndExecuteKernel(compiled_kernel, d_A, d_B, d_C, M, N, K,
                                             grid_x, grid_y, 0);
                    },
                    [&]() {
                        std::vector<float> h_result(M * N);
                        checkCudaErrors(cuMemcpyDtoH(h_result.data(), d_C,
                                                     M * N * sizeof(float)));
                        return verify_matmul_result(config_name.c_str(),
                                                    h_result.data(), h_C.data(), M, N);
                    },
                    M, N, K);
                print_result(result);

                autotune_results.push_back({tile.block_m, tile.block_n, tile.block_k,
                                            grid_x, grid_y, load_lat, store_lat, result});
            }
        }
    }

    // find the best configuration by GFLOPS
    auto best = std::max_element(autotune_results.begin(), autotune_results.end(),
        [](const AutotuneResult& a, const AutotuneResult& b) {
            return a.result.gflops < b.result.gflops;
        });

    printf("\n  *** BEST CONFIGURATION ***\n");
    printf("  BLOCK_M=%d, BLOCK_N=%d, BLOCK_K=%d\n",
           best->block_m, best->block_n, best->block_k);
    printf("  LOAD_LATENCY=%d, STORE_LATENCY=%d, grid_x=%d, grid_y=%d\n",
           best->load_latency, best->store_latency, best->grid_x, best->grid_y);
    printf("  Performance: %.1f GFLOPS, %.3f ms, %.1f GB/s\n",
           best->result.gflops, best->result.time_ms, best->result.bandwidth_gb_s);

    checkCudaErrors(cuMemFree(d_A));
    checkCudaErrors(cuMemFree(d_B));
    checkCudaErrors(cuMemFree(d_C));
}

int main(int argc, char** argv) {
    std::vector<char*> benchmark_argv;
    CompilerBackend compiler_backend = parseCompilerBackendArgs(argc, argv, benchmark_argv);
    parse_benchmark_args(static_cast<int>(benchmark_argv.size()), benchmark_argv.data());
    print_device_info();

    // initialize CUDA and get compute capability
    setSMValue();

    CUcontext context;
    CUdevice cuDevice = 0;
    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGet(&cuDevice, 0));
    checkCudaErrors(cuCtxCreate(&context, NULL, 0, cuDevice));

    printf("\nMatrix Multiplication Autotuner (FP16 inputs, FP32 accumulate)\n");
    printf("==============================================================\n");
    printf("Backend: %s\n", compilerBackendName(compiler_backend));

    char *kernel_file = findSampleFile("matmul.cu", argv[0]);
    if (kernel_file == NULL) {
        fprintf(stderr, "Error: unable to locate matmul.cu\n");
        return 1;
    }

    char *search_space_file = findSampleFile(kSearchSpaceFileName, argv[0]);
    if (search_space_file == NULL) {
        fprintf(stderr, "Error: unable to locate %s\n", kSearchSpaceFileName);
        free(kernel_file);
        return 1;
    }

    SearchSpace search_space = loadSearchSpace(search_space_file);
    printf("Search space: %s\n", search_space_file);

    printf("Tuning for M=1024, N=4096, K=1024\n");
    autotuner(1024, 4096, 1024, kernel_file, search_space, compiler_backend);
    free(kernel_file);
    free(search_space_file);

    return 0;
}
