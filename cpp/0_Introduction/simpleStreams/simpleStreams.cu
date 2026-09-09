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
 * This sample illustrates the usage of CUDA streams for overlapping
 * kernel execution with device/host memcopies.  The kernel computes an
 * element-wise square of a float array, after which the result is copied
 * back to host (CPU) memory.  To increase performance, multiple
 * kernel/memcopy pairs are launched asynchronously, each pair in its
 * own stream.  A GPU can overlap a kernel and a memcopy as long as they
 * are issued in different streams.  Thus, if n pairs are launched, the
 * streamed approach can reduce the memcopy cost to the (1/n)th of a
 * single copy of the entire data set.
 *
 * Additionally, this sample uses CUDA events to measure elapsed time for
 * CUDA calls.  Events are a part of CUDA API and provide a system independent
 * way to measure execution times on CUDA devices with approximately 0.5
 * microsecond precision.
 *
 */

// System includes
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

#define N (1 << 24) // ~16M elements
#define BLOCK 256   // threads per block
#define NUM_STREAMS 4   // number of concurrent streams

// Kernel: computes element-wise square of the input array
__global__ void square_kernel(const float* in, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        out[idx] = in[idx] * in[idx];
    }
}

// Run the full H2D memcopy + kernel + D2H memcopy in a single default stream.
// Returns elapsed time in milliseconds.
float run_default_stream(float* h_in, float* h_out, int n){
    float *d_in, *d_out;  // device input and output buffers
    size_t bytes = n * sizeof(float);

    // allocate device memory
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // create CUDA event handles for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // copy input to device, run kernel, copy result back
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    int grid = (n + BLOCK - 1) / BLOCK;
    square_kernel<<<grid, BLOCK>>>(d_in, d_out, n);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // block until the event is actually recorded

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // release device resources
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// Run the same workload split across 4 streams so that H2D memcopies,
// kernel execution, and D2H memcopies can overlap on the device.
// Returns elapsed time in milliseconds.
float run_multi_stream(float* h_in, float* h_out, int n){
    int chunk = n/NUM_STREAMS;                      // number of elements per stream
    size_t chunk_bytes = chunk * sizeof(float);

    float *d_in[NUM_STREAMS], *d_out[NUM_STREAMS]; // per-stream device buffers
    cudaStream_t streams[NUM_STREAMS];

    // allocate and initialize an array of stream handles
    for(int i = 0; i < NUM_STREAMS; i++){
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        cudaMalloc(&d_in[i], chunk_bytes);
        cudaMalloc(&d_out[i], chunk_bytes);
    }

    // create CUDA event handles for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, streams[0]);

    // asynchronously launch 4 copies + kernels + copies, each in its own
    // stream so they can overlap on the device
    for(int i = 0; i < NUM_STREAMS; i++){
        int offset = i * chunk; // element offset into the host arrays for this stream

        // H2D: copy this stream's chunk to device (non-blocking on host)
        cudaMemcpyAsync(
            d_in[i],
            h_in + offset,
            chunk_bytes,
            cudaMemcpyHostToDevice,
            streams[i]
        );

        // kernel: will only start after the H2D copy in this stream completes
        int grid = (chunk + BLOCK - 1) / BLOCK;
        square_kernel<<<grid, BLOCK, 0, streams[i]>>>(
            d_in[i], d_out[i], chunk
        );

        // D2H: copy result back; starts after the kernel in this stream finishes
        cudaMemcpyAsync(
            h_out + offset,
            d_out[i],
            chunk_bytes,
            cudaMemcpyDeviceToHost,
            streams[i]
        );
    }

    // Wait for all streams to complete their work
    for(int i=0; i < NUM_STREAMS; i++) { 
       cudaStreamSynchronize(streams[i]);
     }
    cudaEventRecord(stop, streams[0]);

    // Sync with stream 0, which only completes once all other work
    // is done and the stop event is recorded
    cudaStreamSynchronize(streams[0]);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // release per-stream resources
    for(int i=0; i < NUM_STREAMS; i++){
        cudaFree(d_in[i]);
        cudaFree(d_out[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main(){

    printf("[ CUDA Sample: Streams ]\n\n");

    // Query compute capability and number of SMs on device 0
    int cuda_device = 0;
    int major = 0, minor = 0, smCount = 0;
    cudaDeviceGetAttribute(&major,   cudaDevAttrComputeCapabilityMajor, cuda_device);
    cudaDeviceGetAttribute(&minor,   cudaDevAttrComputeCapabilityMinor, cuda_device);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount,    cuda_device);
    printf("GPU Device %d: with compute capability %d.%d and Number of SMs %d\n\n",
           cuda_device, major, minor, smCount);

    size_t bytes = N * sizeof(float);

    // allocate pinned host memory: the OS normally moves RAM pages around freely,
    // pinning locks these pages in place so the GPU can read them directly
    // without the data being moved away mid-transfer
    float *h_in  = 0; // pointer to input data in host memory
    float *h_out = 0; // pointer to output data in host memory
    cudaMallocHost(&h_in, bytes);
    cudaMallocHost(&h_out, bytes);

    // initialize input array
    for (int i = 0; i < N; i++)
        h_in[i] = (float)i;

    //////////////////////////////////////////////////////////////////////
    // time single-stream execution for reference
    float t1 = run_default_stream(h_in, h_out, N);

    //////////////////////////////////////////////////////////////////////
    // time execution with 4 streams
    float t2 = run_multi_stream(h_in, h_out, N);

    printf("Single stream = %.3f ms\n", t1);
    printf("Multi-stream  = %.3f ms\n", t2);
    printf("Speedup       = %.2fx\n",   t1 / t2);

    // release pinned host memory
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);

    return 0;
}
