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
 * simpleCallback
 * --------------
 * An example of a heterogeneous pipeline in the form:
 *
 *      CPU pre-process  ->  GPU kernel  ->  CPU post-process
 *
 * The whole pipeline is coordinated by a *single* CUDA stream. A stream
 * executes the operations enqueued on it in order, so we can simply push work
 * onto it and let CUDA take care of ordering and dependencies for us.
 *
 * Two different "CPU" mechanisms appear in this sample so you can see how they
 * differ:
 *
 *    1. The CPU pre-processing runs on a dedicated worker thread created with
 *      C++ std::thread. This shows how ordinary host-side threading can be
 *      combined with CUDA. The worker fills the input buffer and then enqueues
 *      the GPU pipeline itself; main joins it and waits on the stream.
 *
 *   2. The CPU post-processing is scheduled *on the stream itself* with
 *      cudaLaunchHostFunc(). CUDA calls our host function automatically once all
 *      the preceding stream work (copies + kernel) has completed. This is the
 *      modern, non-deprecated replacement for cudaStreamAddCallback().
 *
 * IMPORTANT rule for cudaLaunchHostFunc callbacks:
 *   The host function must NOT call any CUDA runtime/driver API (no cudaMalloc,
 *   cudaFree, kernel launches, etc.). It should only do plain CPU work.
 *
 * NOTE: For clarity this sample intentionally omits CUDA error checking. Real
 * applications should check the return code of every CUDA call.
 */

// System includes
#include <cstdio>
#include <cstdlib>

#include <thread>    // std::thread — C++11 standard threading

// CUDA runtime
#include <cuda_runtime.h>

const int NumElements = 100000;

// Everything a single pipeline needs. Passed by pointer to both the 
// std::thread worker and to the stream host-function callback.
struct Workload
{
    int          id      = 0;       // arbitrary tag added to each element
    int         *h_data  = nullptr; // pinned host buffer (fast async copies)
    int         *d_data  = nullptr; // device buffer
    cudaStream_t stream  = nullptr; // the single stream that orders all the work
    bool         success = false;   // set by the post-processing callback
};

// GPU kernel: increment every element by one.
__global__ void incrementKernel(int *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] += 1;
    }
}

// Forward declaration: postprocess is defined below but called from preprocess.
void CUDART_CB postprocess(void *arg);

// Worker thread: CPU pre-processing + full GPU pipeline enqueue.
//
// std::thread passes arguments with their actual types, so no void* casting
// is needed. This thread fills the input buffer, then enqueues the full GPU
// pipeline (H2D, kernel, D2H, host func) onto the workload's stream.
void preprocess(Workload *workload)
{
    // Stage 1: CPU pre-processing ----------------------------------
    printf("[thread]    Stage 1: CPU pre-processing on a worker thread...\n");
    for (int i = 0; i < NumElements; ++i) {
        workload->h_data[i] = workload->id + i;
    }
    printf("[thread]    Filled %d elements. First 3 inputs: %d, %d, %d\n\n",
           NumElements, workload->h_data[0], workload->h_data[1], workload->h_data[2]);

    // Stage 2: enqueue GPU work from this thread -------------------
    const size_t  bytes          = NumElements * sizeof(int);
    const int     threadsPerBlock = 256;
    const int     blocks          = (NumElements + threadsPerBlock - 1) / threadsPerBlock;

    printf("[thread]    Stage 2: enqueuing H2D copy, kernel, D2H copy on the stream\n");
    cudaMemcpyAsync(workload->d_data, workload->h_data, bytes, cudaMemcpyHostToDevice, workload->stream);
    incrementKernel<<<blocks, threadsPerBlock, 0, workload->stream>>>(workload->d_data, NumElements);
    cudaMemcpyAsync(workload->h_data, workload->d_data, bytes, cudaMemcpyDeviceToHost, workload->stream);

    // Stage 3: schedule CPU post-processing on the stream -----------
    printf("[thread]    Registering post-processing callback with cudaLaunchHostFunc\n\n");
    cudaLaunchHostFunc(workload->stream, postprocess, workload);
}

// Stage 3: CPU post-processing, scheduled on the stream via cudaLaunchHostFunc.
//
// CUDA invokes this automatically once the H2D copy, the kernel, and the D2H
// copy that precede it on the stream have all finished. Every element should
// now equal its original value plus one (from the kernel).
//
// Reminder: DO NOT call any CUDA API from inside this function.
void CUDART_CB postprocess(void *arg)
{
    // recover the typed pointer from the callback's void* arg
    auto *workload = static_cast<Workload *>(arg);

    // Stage 3: CPU post-processing — runs automatically when all stream work above is done
    printf("[host func] Stage 3: callback fired automatically - the GPU work is done!\n");
    printf("[host func] First 3 results: %d, %d, %d (each input +1)\n",
           workload->h_data[0], workload->h_data[1], workload->h_data[2]);

    bool verify = true;
    for (int i = 0; i < NumElements; ++i) {
        if (workload->h_data[i] != workload->id + i + 1) {
            verify = false;
            break;
        }
    }
    workload->success = verify;
    printf("[host func] Verified all %d results: %s\n", NumElements, verify ? "PASS" : "MISMATCH");
}

int main()
{
    printf("=====================================================\n");
    printf("  simpleCallback: CPU -> GPU -> CPU pipeline demo\n");
    printf("=====================================================\n");

    // Use the first available CUDA device.
    int devID = 0;
    cudaSetDevice(devID);

    int major = 0, minor = 0, smCount = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, devID);
    printf("Using GPU %d: compute capability %d.%d, %d SMs\n", devID, major, minor, smCount);

    // Allocate resources 
    Workload workload;
    workload.id = 42; // Any number can be used

    // Pinned (page-locked) host memory enables true asynchronous copies.
    const size_t bytes = NumElements * sizeof(int);
    cudaMallocHost(&workload.h_data, bytes);
    cudaMalloc(&workload.d_data, bytes);
    cudaStreamCreate(&workload.stream);

    // Launch worker thread; it fills the buffer AND enqueues the full GPU pipeline.
    std::thread worker(preprocess, &workload);
    worker.join();
    printf("[main]      Worker thread joined; GPU work has been enqueued.\n");

    // Block until the whole pipeline (including the host function) is complete.
    printf("[main]      Waiting for the stream (and callback) to finish...\n\n");
    cudaStreamSynchronize(workload.stream);

    // Clean up
    // Safe to call CUDA APIs here: we are back on the main thread, not inside
    // the host-function callback.
    cudaStreamDestroy(workload.stream);
    cudaFree(workload.d_data);
    cudaFreeHost(workload.h_data);

    printf("\n[main]      Pipeline complete. Result: %s\n", workload.success ? "SUCCESS" : "FAILURE");
    return workload.success ? EXIT_SUCCESS : EXIT_FAILURE;
}
