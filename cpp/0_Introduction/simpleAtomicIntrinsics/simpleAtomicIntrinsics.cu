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

/* A simple program demonstrating why atomic operations are needed, using the
 * atomic intrinsics atomicAdd, atomicMax, and atomicCAS.
 *
 * 1,000,000 threads write into an array of only 10 integers (each thread maps
 * to a slot with index % ARRAY_SIZE), so ~100,000 threads update every element
 * at the same time. A plain update like g[i] = g[i] + 1 is three separate
 * steps: read the value, modify it, write it back. When two threads interleave
 * (both read the same old value, both write back the same result), one update
 * silently overwrites the other and is lost. This is a race condition: the
 * result depends on thread timing, not program logic, and changes every run.
 *
 * Each operation is therefore run four ways: a non-atomic kernel that loses
 * most of its updates to these race conditions, and three atomic kernels that
 * do the read-modify-write as one indivisible operation and give the exact
 * result — first with the CUDA atomic intrinsics, then with
 * cuda::std::atomic_ref from CCCL (the std::atomic API in device code), and
 * finally with cuda::atomic_ref from CCCL (the CUDA-specific variant that
 * natively supports CUDA thread scopes).
 */

// includes, system
#include <stdio.h>

// Includes CUDA
#include <cuda_runtime.h>

// Includes CCCL: std::atomic as usable in device code (cuda::std::atomic_ref)
#include <cuda/std/atomic>

// cuda::atomic_ref<T>: like cuda::std::atomic_ref but CUDA-specific, supports thread scopes
#include <cuda/atomic>

// and cuda::ceil_div for computing the launch grid size
#include <cuda/cmath>


// Launch configuration: many threads hammering a small array
#define NUM_THREADS 1000000
#define BLOCK_WIDTH 1000
#define ARRAY_SIZE  10

// Increment without atomics: threads race on shared elements and lose updates
__global__ void increment(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // ceil_div rounds up the grid, this may launch extra threads
    // Adding this guard to ignore extra threads
    if (tid < NUM_THREADS) {
        // each thread increments one element, wrapping at ARRAY_SIZE
        int i = tid % ARRAY_SIZE;
        g[i]  = g[i] + 1;
    }
}

// Increment with atomicAdd: read-modify-write is one indivisible operation
__global__ void increment_atomic(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;
        atomicAdd(&g[i], 1);
    }
}

// Increment with cuda::std::atomic_ref: wraps the plain int in g[] and exposes
// the exact std::atomic API (fetch_add, load, store, ...) inside device code
__global__ void increment_atomic_std(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;
        cuda::std::atomic_ref<int> ref(g[i]);
        ref.fetch_add(1);
    }
}

// Increment with cuda::atomic_ref: like cuda::std::atomic_ref but CUDA-specific
// and natively supports CUDA thread scopes
__global__ void increment_atomic_cuda(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;
        cuda::atomic_ref<int> ref(g[i]);
        ref.fetch_add(1);
    }
}

// Max without atomics: another thread can write between the compare and the
// store, so a smaller value can overwrite a larger one
__global__ void max(int *g)
{
    // each thread contributes its global index as the value
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;
        if (g[i] < tid)
            g[i] = tid;
    }
}

// Max with atomicMax: the compare and the store happen as one operation
__global__ void max_atomic(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;
        atomicMax(&g[i], tid);
    }
}

// Max with cuda::std::atomic_ref: std::atomic has no fetch_max, so max is
// built the standard C++ way — a compare_exchange loop that stops as soon as
// the stored value is already >= ours
__global__ void max_atomic_std(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;

        cuda::std::atomic_ref<int> ref(g[i]);
        int expected = ref.load();
        while (expected < tid && !ref.compare_exchange_weak(expected, tid)) {
        }
    }
}

// Max with cuda::atomic_ref: same compare_exchange loop as the _std variant
__global__ void max_atomic_cuda(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;

        cuda::atomic_ref<int> ref(g[i]);
        int expected = ref.load();
        while (expected < tid && !ref.compare_exchange_weak(expected, tid)) {
        }
    }
}

// Increment written as compare-and-swap, without atomics: the value can change
// between the compare and the swap, so increments are lost
__global__ void cas(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;

        int expected = g[i];        // read
        if (g[i] == expected)       // compare
            g[i] = expected + 1;    // swap (not atomic with the compare!)
    }
}

// Increment with an atomicCAS retry loop: the classic pattern for building
// any atomic operation out of compare-and-swap. atomicCAS returns the value
// it found: if another thread interfered, the swap did not happen and we retry
__global__ void cas_atomic(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;

        int old = g[i];
        int assumed;
        do {
            assumed = old;
            old     = atomicCAS(&g[i], assumed, assumed + 1);
        } while (old != assumed);
    }
}

// Increment with cuda::std::atomic_ref compare_exchange: the std::atomic way
// to write a CAS retry loop. On failure, expected is updated with the value
// actually found, so the loop just retries with fresh data
__global__ void cas_atomic_std(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;

        cuda::std::atomic_ref<int> ref(g[i]);
        int expected = ref.load();
        while (!ref.compare_exchange_weak(expected, expected + 1)) {
        }
    }
}

// CAS increment with cuda::atomic_ref: same retry loop as the _std variant
__global__ void cas_atomic_cuda(int *g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_THREADS) {
        int i = tid % ARRAY_SIZE;

        cuda::atomic_ref<int> ref(g[i]);
        int expected = ref.load();
        while (!ref.compare_exchange_weak(expected, expected + 1)) {
        }
    }
}

// Print every element of the array
void print_array(int *array, int size)
{
    printf("{ ");
    for (int i = 0; i < size; i++)
        printf("%d ", array[i]);
    printf("}\n");
}

// Program main
int main(int argc, char **argv)
{
    printf("=== Atomic vs. non-atomic operations (intrinsics, cuda::std::atomic_ref, cuda::atomic_ref) ===\n\n");

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

    // enough blocks to cover all threads, rounding up if not evenly divisible
    int numBlocks = cuda::ceil_div(NUM_THREADS, BLOCK_WIDTH);
    printf("%d total threads in %d blocks writing into %d array elements\n\n",
           NUM_THREADS, numBlocks, ARRAY_SIZE);

    // declare and allocate host memory
    int       h_array[ARRAY_SIZE];
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

    // declare and allocate GPU memory (zeroed with cudaMemset before each run)
    int *d_array;
    cudaMalloc((void **)&d_array, ARRAY_BYTES);

    // CUDA events record timestamps on the GPU stream to measure device time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_ms;

    // ----- add: every thread adds 1, each element should reach 100000 -----
    printf("[add] expected: every element = %d\n", NUM_THREADS / ARRAY_SIZE);

    cudaMemset((void *)d_array, 0, ARRAY_BYTES);

    cudaEventRecord(start);
    increment<<<numBlocks, BLOCK_WIDTH>>>(d_array);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    printf("%-36s (%g ms): ", "non-atomic", elapsed_ms);
    print_array(h_array, ARRAY_SIZE);

    cudaMemset((void *)d_array, 0, ARRAY_BYTES);

    cudaEventRecord(start);
    increment_atomic<<<numBlocks, BLOCK_WIDTH>>>(d_array);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    printf("%-36s (%g ms): ", "atomicAdd()", elapsed_ms);
    print_array(h_array, ARRAY_SIZE);

    cudaMemset((void *)d_array, 0, ARRAY_BYTES);

    cudaEventRecord(start);
    increment_atomic_std<<<numBlocks, BLOCK_WIDTH>>>(d_array);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    printf("%-36s (%g ms): ", "cuda::std::atomic_ref::fetch_add()", elapsed_ms);
    print_array(h_array, ARRAY_SIZE);

    cudaMemset((void *)d_array, 0, ARRAY_BYTES);

    cudaEventRecord(start);
    increment_atomic_cuda<<<numBlocks, BLOCK_WIDTH>>>(d_array);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    printf("%-36s (%g ms): ", "cuda::atomic_ref::fetch_add()", elapsed_ms);
    print_array(h_array, ARRAY_SIZE);

    // ----- max: every thread offers its index, element i should reach
    // the largest index that maps to it: NUM_THREADS - ARRAY_SIZE + i -----
    printf("\n[max] expected: element i = %d + i\n", NUM_THREADS - ARRAY_SIZE);

    cudaMemset((void *)d_array, 0, ARRAY_BYTES);

    cudaEventRecord(start);
    max<<<numBlocks, BLOCK_WIDTH>>>(d_array);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    printf("%-46s (%g ms): ", "non-atomic", elapsed_ms);
    print_array(h_array, ARRAY_SIZE);

    cudaMemset((void *)d_array, 0, ARRAY_BYTES);

    cudaEventRecord(start);
    max_atomic<<<numBlocks, BLOCK_WIDTH>>>(d_array);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    printf("%-46s (%g ms): ", "atomicMax()", elapsed_ms);
    print_array(h_array, ARRAY_SIZE);

    cudaMemset((void *)d_array, 0, ARRAY_BYTES);

    cudaEventRecord(start);
    max_atomic_std<<<numBlocks, BLOCK_WIDTH>>>(d_array);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    printf("%-46s (%g ms): ", "cuda::std::atomic_ref::compare_exchange_weak()", elapsed_ms);
    print_array(h_array, ARRAY_SIZE);

    cudaMemset((void *)d_array, 0, ARRAY_BYTES);

    cudaEventRecord(start);
    max_atomic_cuda<<<numBlocks, BLOCK_WIDTH>>>(d_array);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    printf("%-46s (%g ms): ", "cuda::atomic_ref::compare_exchange_weak()", elapsed_ms);
    print_array(h_array, ARRAY_SIZE);

    // ----- CAS: increment built from compare-and-swap, same expected
    // result as add -----
    printf("\n[CAS] expected: every element = %d\n", NUM_THREADS / ARRAY_SIZE);

    cudaMemset((void *)d_array, 0, ARRAY_BYTES);

    cudaEventRecord(start);
    cas<<<numBlocks, BLOCK_WIDTH>>>(d_array);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    printf("%-46s (%g ms): ", "non-atomic", elapsed_ms);
    print_array(h_array, ARRAY_SIZE);

    cudaMemset((void *)d_array, 0, ARRAY_BYTES);

    cudaEventRecord(start);
    cas_atomic<<<numBlocks, BLOCK_WIDTH>>>(d_array);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    printf("%-46s (%g ms): ", "atomicCAS()", elapsed_ms);
    print_array(h_array, ARRAY_SIZE);

    cudaMemset((void *)d_array, 0, ARRAY_BYTES);

    cudaEventRecord(start);
    cas_atomic_std<<<numBlocks, BLOCK_WIDTH>>>(d_array);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    printf("%-46s (%g ms): ", "cuda::std::atomic_ref::compare_exchange_weak()", elapsed_ms);
    print_array(h_array, ARRAY_SIZE);

    cudaMemset((void *)d_array, 0, ARRAY_BYTES);

    cudaEventRecord(start);
    cas_atomic_cuda<<<numBlocks, BLOCK_WIDTH>>>(d_array);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    printf("%-46s (%g ms): ", "cuda::atomic_ref::compare_exchange_weak()", elapsed_ms);
    print_array(h_array, ARRAY_SIZE);

    printf("\nThe non-atomic kernels lose updates when threads race on the same\n");
    printf("element; the atomic kernels match the expected values exactly.\n");

    // free GPU memory allocation and timing events, then exit
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_array);
    return 0;
}
