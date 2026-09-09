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
 * This example shows how to use the clock function to measure the performance
 * of block of threads of a kernel accurately. Blocks are executed in parallel
 * and out of order. Since there's no synchronization mechanism between blocks,
 * we measure the clock once for each block. The clock samples are written to
 * device memory.
 */

// System includes
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUB for block-scope reduction
#include <cub/cub.cuh>

#define NUM_BLOCKS  64
#define NUM_THREADS 256

// This kernel computes a standard parallel reduction and evaluates the
// time it takes to do that for each block. The timing results are stored
// in device memory.
__global__ static void timedReduction(const float *input, float *output, clock_t *timer)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid == 0)
        timer[bid] = clock();

    // Each thread loads 2 elements and reduces them to a local min.
    float thread_data[2];
    thread_data[0] = input[tid];
    thread_data[1] = input[tid + blockDim.x];

    // Block-wide min-reduction using CUB. Default constructor allocates
    // shared memory internally via PrivateStorage().
    using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
    float block_min = BlockReduce().Reduce(thread_data, [] __device__(float a, float b) { return fminf(a, b); });

    // Only thread 0 holds the valid aggregate.
    if (tid == 0)
        output[bid] = block_min;

    __syncthreads();

    if (tid == 0)
        timer[bid + gridDim.x] = clock();
}

// Start the main CUDA Sample here
int main(int argc, char **argv)
{
    printf("CUDA Clock sample\n");

    // Select device 0 as the active GPU
    int devID = 0;
    cudaSetDevice(devID);

    // Query compute capability (major.minor) and number of SMs on the device
    int major = 0, minor = 0, smCount = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, devID);

    // Print device info
    printf("GPU Device %d: with compute capability %d.%d and Number of SMs %d\n\n", devID, major, minor, smCount);

    // Device pointers for input data, per-block minimum output, and clock timestamps
    float   *dinput  = NULL;
    float   *doutput = NULL;
    clock_t *dtimer  = NULL;

    clock_t timer[NUM_BLOCKS * 2];
    float   input[NUM_THREADS * 2];

    for (int i = 0; i < NUM_THREADS * 2; i++) {
        input[i] = (float)i;
    }

    cudaMalloc((void **)&dinput, sizeof(float) * NUM_THREADS * 2);
    cudaMalloc((void **)&doutput, sizeof(float) * NUM_BLOCKS);
    cudaMalloc((void **)&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2);

    cudaMemcpy(dinput, input, sizeof(float) * NUM_THREADS * 2, cudaMemcpyHostToDevice);

    timedReduction<<<NUM_BLOCKS, NUM_THREADS>>>(dinput, doutput, dtimer);

    cudaMemcpy(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2, cudaMemcpyDeviceToHost);

    cudaFree(dinput);
    cudaFree(doutput);
    cudaFree(dtimer);

    long double avgElapsedClocks = 0;

    for (int i = 0; i < NUM_BLOCKS; i++) {
        avgElapsedClocks += (long double)(timer[i + NUM_BLOCKS] - timer[i]);
    }

    avgElapsedClocks = avgElapsedClocks / NUM_BLOCKS;
    printf("Average clocks/block = %Lf\n", avgElapsedClocks);

    return EXIT_SUCCESS;
}
