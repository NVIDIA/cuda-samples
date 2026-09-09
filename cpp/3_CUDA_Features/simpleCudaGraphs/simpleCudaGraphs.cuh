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
 * Shared code for the simpleCudaGraphs sample. Both the explicit-graph and
 * stream-capture demos build a graph around the same two-pass reduction, so the
 * reduction kernels, the host callback and its data type, and the input-fill
 * helper live here to avoid duplicating them across the two .cu files.
 */

#pragma once

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define THREADS_PER_BLOCK 512

// Payload passed to the host-callback node: the demo name and the reduced result.
typedef struct callBackData
{
    const char *fn_name;
    double     *data;
} callBackData_t;

// Host-callback node body: prints the reduced result, then resets it for the next launch.
inline void CUDART_CB myHostNodeCallback(void *data)
{
    callBackData_t *tmp    = (callBackData_t *)(data);
    double         *result = (double *)(tmp->data);
    printf("[%s] Host callback final reduced sum = %lf\n", tmp->fn_name, *result);
    *result = 0.0; // reset the result
}

// Fill the host input buffer with fresh pseudo-random values. Called before each
// graph launch so the same graph processes different data every iteration.
inline void init_input(float *inputVec, size_t size)
{
    for (size_t i = 0; i < size; i++)
        inputVec[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

// Pass 1: reduce the input float vector to one partial sum per block.
__global__ void reduce(float *inputVec, double *outputVec, size_t inputSize, size_t outputSize)
{
    typedef cub::BlockReduce<double, THREADS_PER_BLOCK> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread adds its assigned elements into a local partial sum
    double temp_sum = 0.0;
    for (size_t i = globaltid; i < inputSize; i += (size_t)gridDim.x * blockDim.x)
        temp_sum += (double)inputVec[i];

    // CUB reduces all per-thread partial sums to a single block sum; result lands on thread 0
    double block_sum = BlockReduceT(temp_storage).Sum(temp_sum);

    if (threadIdx.x == 0 && blockIdx.x < outputSize)
        outputVec[blockIdx.x] = block_sum;
}

// Pass 2: reduce the per-block partial sums to a single scalar result.
__global__ void reduceFinal(double *inputVec, double *result, size_t inputSize)
{
    typedef cub::BlockReduce<double, THREADS_PER_BLOCK> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread adds its assigned elements into a local partial sum
    double temp_sum = 0.0;
    for (size_t i = globaltid; i < inputSize; i += (size_t)gridDim.x * blockDim.x)
        temp_sum += (double)inputVec[i];

    // CUB reduces all per-thread partial sums to a single block sum; result lands on thread 0
    double block_sum = BlockReduceT(temp_storage).Sum(temp_sum);

    if (threadIdx.x == 0)
        result[0] = block_sum;
}
