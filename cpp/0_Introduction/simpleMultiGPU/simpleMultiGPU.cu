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
 * Sums a large vector across every GPU in the system.
 *
 * The input is split into one slice per GPU, each with its own stream, so the H2D copy,
 * reduction kernel, and D2H copy run concurrently across devices. cub::BlockReduce reduces
 * within each thread block; the host then adds the per-block values into a per-GPU total and
 * combines those into the final sum, checked against a CPU reference. CUDA events time the
 * GPU phase.
 */

// System includes
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// CUDA runtime
#include <cuda_runtime.h>

// CCCL / CUB block-wide reduction
#include <cub/cub.cuh>

// Everything one GPU needs: its slice of the input, its buffers, and its stream
struct TGPUplan
{
    int          dataN;             // elements assigned to this GPU
    float       *h_Data;            // pinned input slice
    float       *d_Data;            // the slice, on the device
    float       *d_Sum;             // one partial sum per block
    float       *h_Sum_from_device; // those partial sums, copied back
    cudaStream_t stream;            // orders this GPU's copies and kernel
};

// Data configuration
constexpr int DATA_N = 1048576 * 32;

// Reduction launch configuration
constexpr int BLOCK_N  = 32;  // thread blocks launched per GPU
constexpr int THREAD_N = 256; // threads per block

// Per-block reduction kernel.
// Threads walk the input in a grid-stride loop, then cub::BlockReduce combines their totals.
// Thread 0 writes one partial sum per block (BLOCK_N values total); the host
// adds those up to obtain this GPU's final sum.
__global__ static void reduceKernel(float *d_Result, const float *d_Input, int N)
{
    using BlockReduce = cub::BlockReduce<float, THREAD_N>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadN = gridDim.x * blockDim.x;
    float     sum     = 0;

    for (int pos = tid; pos < N; pos += threadN)
        sum += d_Input[pos];

    // Reduce the per-thread partial sums across the block (result valid in thread 0)
    sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0)
        d_Result[blockIdx.x] = sum;
}

int main()
{
    printf("Starting simpleMultiGPU\n");

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    printf("CUDA-capable device count: %i\n", deviceCount);

    // This sample needs at least two GPUs to show work running on several devices at once
    if (deviceCount < 2) {
        printf("Two or more CUDA-capable devices are required. Exiting.\n");
        return EXIT_SUCCESS;
    }

    printf("Generating input data...\n\n");

    std::vector<TGPUplan> plan(deviceCount);
    std::vector<float>    h_SumGPU(deviceCount); // one result per GPU

    // Subdividing input data across GPUs
    // Get data sizes for each GPU
    for (int i = 0; i < deviceCount; i++) {
        plan[i].dataN = DATA_N / deviceCount;
    }

    // Take into account "odd" data sizes
    for (int i = 0; i < DATA_N % deviceCount; i++) {
        plan[i].dataN++;
    }

    // Create streams for issuing GPU command asynchronously and allocate memory
    // (GPU and System page-locked)
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&plan[i].stream);

        cudaMalloc(&plan[i].d_Data, plan[i].dataN * sizeof(float));
        cudaMalloc(&plan[i].d_Sum, BLOCK_N * sizeof(float));
        cudaMallocHost(&plan[i].h_Sum_from_device, BLOCK_N * sizeof(float));
        cudaMallocHost(&plan[i].h_Data, plan[i].dataN * sizeof(float));

        for (int j = 0; j < plan[i].dataN; j++) {
            plan[i].h_Data[j] = (float)rand() / (float)RAND_MAX;
        }
    }

    // Start timing and compute on GPU(s)
    printf("Computing with %d GPUs...\n", deviceCount);
    // Time the multi-GPU work with CUDA events (recorded on device 0)
    cudaEvent_t start, stop;
    cudaSetDevice(0);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Queue every GPU's copy-in, kernel and copy-out before waiting on any of them,
    // so the devices run concurrently instead of one after another
    for (int i = 0; i < deviceCount; i++) {
        TGPUplan &gpu = plan[i];
        cudaSetDevice(i);

        // Copy input data from CPU
        cudaMemcpyAsync(gpu.d_Data, gpu.h_Data, gpu.dataN * sizeof(float), cudaMemcpyHostToDevice, gpu.stream);

        // Launch Kernel
        reduceKernel<<<BLOCK_N, THREAD_N, 0, gpu.stream>>>(gpu.d_Sum, gpu.d_Data, gpu.dataN);

        // Read back GPU results
        cudaMemcpyAsync(gpu.h_Sum_from_device, gpu.d_Sum, BLOCK_N * sizeof(float), cudaMemcpyDeviceToHost, gpu.stream);
    }

    // Process GPU results
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);

        // Wait for all operations to finish
        cudaStreamSynchronize(plan[i].stream);

        // Finalize GPU reduction for current subvector
        float sum = 0;

        for (int j = 0; j < BLOCK_N; j++) {
            sum += plan[i].h_Sum_from_device[j];
        }

        h_SumGPU[i] = sum;

        // Free up this GPU's resources. h_Data stays alive for the CPU check below.
        cudaFreeHost(plan[i].h_Sum_from_device);
        cudaFree(plan[i].d_Sum);
        cudaFree(plan[i].d_Data);
        cudaStreamDestroy(plan[i].stream);
    }

    float sumGPU = 0;

    for (int i = 0; i < deviceCount; i++) {
        sumGPU += h_SumGPU[i];
    }

    cudaSetDevice(0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("  GPU Processing time: %f (ms)\n\n", gpuTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Compute on Host CPU
    printf("Computing with Host CPU...\n\n");

    double sumCPU = 0;

    for (int i = 0; i < deviceCount; i++) {
        for (int j = 0; j < plan[i].dataN; j++) {
            sumCPU += plan[i].h_Data[j];
        }
    }

    // Compare GPU and CPU results
    printf("Comparing GPU and Host CPU results...\n");
    const double diff = fabs(sumCPU - sumGPU) / fabs(sumCPU);
    printf("  GPU sum: %f\n  CPU sum: %f\n", sumGPU, sumCPU);
    printf("  Relative difference: %E \n\n", diff);

    // Cleanup and shutdown
    for (int i = 0; i < deviceCount; i++) {
        cudaFreeHost(plan[i].h_Data);
    }

    return (diff < 1e-5) ? EXIT_SUCCESS : EXIT_FAILURE;
}
