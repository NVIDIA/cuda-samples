/* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

/* This sample demonstrates the random-number facilities added to libcu++
 * in CCCL 3.3: <cuda/std/random> now offers host- and device-compatible
 * implementations of the standard C++ distributions (uniform, normal,
 * Poisson, Bernoulli, ...), and backports the C++26 Philox counter-based
 * engines. <cuda/random> adds cuda::pcg64 as an NVIDIA extension (the
 * same generator NumPy uses by default).
 *
 * A kernel draws many samples from four different distributions on each
 * thread and the host computes empirical summary statistics, comparing
 * them to the theoretical mean / variance / probability.
 */

/* Includes, system */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <helper_cuda.h>

/* Includes, cccl */
#include <cuda/random>
#include <cuda/std/random>

#define THREADS_PER_BLOCK  256
#define SAMPLES_PER_THREAD 256

/* Per-thread kernel: seed a PCG engine, draw samples from four
 * distributions, and also pull Philox output through a Bernoulli dist
 * to show that distributions work with any engine. */
__global__ void sample_kernel(unsigned long long base_seed,
                              int                num_samples_per_thread,
                              float             *uniform_out,
                              float             *normal_out,
                              int               *poisson_out,
                              int               *bernoulli_out)
{
    const int tid           = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    cuda::pcg64 rng(base_seed + static_cast<unsigned long long>(tid));

    cuda::std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
    cuda::std::normal_distribution<float>       normal_dist(0.0f, 1.0f);
    cuda::std::poisson_distribution<int>        poisson_dist(4.0);
    cuda::std::bernoulli_distribution           bernoulli_dist(0.25);

    cuda::std::philox4x32 philox(static_cast<cuda::std::uint32_t>(base_seed + 17u + tid));

    for (int i = 0; i < num_samples_per_thread; ++i) {
        const int idx      = i * total_threads + tid;
        uniform_out[idx]   = uniform_dist(rng);
        normal_out[idx]    = normal_dist(rng);
        poisson_out[idx]   = poisson_dist(rng);
        bernoulli_out[idx] = bernoulli_dist(philox) ? 1 : 0;
    }
}

template <typename T>
static void summarize(const std::vector<T> &samples, double expected_mean, double expected_var, const char *label)
{
    const size_t n   = samples.size();
    double       sum = 0.0;
    for (const auto v : samples)
        sum += static_cast<double>(v);
    const double mean = sum / static_cast<double>(n);
    double       sq   = 0.0;
    for (const auto v : samples) {
        const double d = static_cast<double>(v) - mean;
        sq += d * d;
    }
    const double var = sq / static_cast<double>(n - 1);
    printf("%-24s n=%zu  mean=%.4f (exp %.4f)   var=%.4f (exp %.4f)\n",
           label,
           n,
           mean,
           expected_mean,
           var,
           expected_var);
}

static void summarize_bernoulli(const std::vector<int> &samples, double expected_p)
{
    long long ones = 0;
    for (int v : samples)
        ones += v;
    const double p = static_cast<double>(ones) / static_cast<double>(samples.size());
    printf("%-24s n=%zu  p(1)=%.4f (exp %.4f)\n", "bernoulli(0.25):", samples.size(), p, expected_p);
}

int main(int argc, char **argv)
{
    int num_blocks = 64;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--blocks") == 0)
            num_blocks = atoi(argv[i + 1]);
    }
    if (num_blocks <= 0)
        num_blocks = 1;

    int devID = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    printf("Device: %s (Compute Capability %d.%d)\n\n", props.name, props.major, props.minor);

    const int    total_threads = num_blocks * THREADS_PER_BLOCK;
    const size_t n             = static_cast<size_t>(total_threads) * SAMPLES_PER_THREAD;
    printf("Drawing %zu samples per distribution (%d blocks x %d threads x %d samples/thread)\n\n",
           n,
           num_blocks,
           THREADS_PER_BLOCK,
           SAMPLES_PER_THREAD);

    float *d_uniform   = nullptr;
    float *d_normal    = nullptr;
    int   *d_poisson   = nullptr;
    int   *d_bernoulli = nullptr;
    checkCudaErrors(cudaMalloc(&d_uniform, n * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_normal, n * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_poisson, n * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_bernoulli, n * sizeof(int)));

    const unsigned long long seed = 0xC0FFEE00ULL;
    sample_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        seed, SAMPLES_PER_THREAD, d_uniform, d_normal, d_poisson, d_bernoulli);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<float> uniform(n), normal(n);
    std::vector<int>   poisson(n), bernoulli(n);
    checkCudaErrors(cudaMemcpy(uniform.data(), d_uniform, n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(normal.data(), d_normal, n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(poisson.data(), d_poisson, n * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(bernoulli.data(), d_bernoulli, n * sizeof(int), cudaMemcpyDeviceToHost));

    summarize(uniform, /*mean=*/0.5, /*var=*/1.0 / 12.0, "uniform(0,1):");
    summarize(normal, /*mean=*/0.0, /*var=*/1.0, "normal(0,1):");
    summarize(poisson, /*mean=*/4.0, /*var=*/4.0, "poisson(lambda=4):");
    summarize_bernoulli(bernoulli, /*p=*/0.25);

    printf("\nEngines exercised: cuda::pcg64 (NumPy-compatible) and cuda::std::philox4x32 (C++26)\n");

    checkCudaErrors(cudaFree(d_uniform));
    checkCudaErrors(cudaFree(d_normal));
    checkCudaErrors(cudaFree(d_poisson));
    checkCudaErrors(cudaFree(d_bernoulli));

    printf("Done\n");
    return EXIT_SUCCESS;
}
