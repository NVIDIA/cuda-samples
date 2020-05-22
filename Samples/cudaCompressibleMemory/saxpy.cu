/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

//
// This sample uses the compressible memory allocation if device supports it
// and performs saxpy on it. 
// Compressible memory may give better performance if the data is amenable to 
// compression.

#include <stdio.h>
#include <cuda.h>
#define CUDA_DRIVER_API
#include "helper_cuda.h"
#include "compMalloc.h"

__global__ void saxpy(float a, float4 *x, float4 *y, float4 *z, int64_t n)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    z[i] = make_float4(a * x[i].x + y[i].x,
                       a * x[i].y + y[i].y,
                       a * x[i].z + y[i].z,
                       a * x[i].w + y[i].w);
}

__global__ void init(float4 *x, float4 *y, float4 *z, float val, int64_t n)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        x[i] = make_float4(val, val, val, val);
        y[i] = make_float4(val, val, val, val);
        z[i] = make_float4(val, val, val, val);
    }
}

int main(int argc, char **argv)
{
    int devId, UseCompressibleMemory = 1;
    int64_t n = 10485760;

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
            checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -UseCompressibleMemory=0 or 1 (default is 1 : Use compressible memory)\n");
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "UseCompressibleMemory")) {
        UseCompressibleMemory = getCmdLineArgumentInt(argc, (const char **)argv, "UseCompressibleMemory");
        if (UseCompressibleMemory > 1) {
            printf("Permitted options for UseCompressibleMemory are 0 or 1, you have entered %d \n", UseCompressibleMemory);
            exit(EXIT_WAIVED);
        }
    }

    devId = findCudaDevice(argc, (const char**)argv);
    CUdevice currentDevice;
    checkCudaErrors(cuCtxGetDevice(&currentDevice));

    // Check that the selected device supports virtual address management
    int vam_supported = -1;
    checkCudaErrors(cuDeviceGetAttribute(&vam_supported,
                          CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                          currentDevice));
    printf("Device %d VIRTUAL ADDRESS MANAGEMENT SUPPORTED = %d.\n", currentDevice, vam_supported);
    if (vam_supported == 0) {
        printf("Device %d doesn't support VIRTUAL ADDRESS MANAGEMENT, so not using compressible memory.\n", currentDevice);
        UseCompressibleMemory = 0;
    }

    int nsm = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, devId)); 
    printf("Found %d SMs on the device\n", nsm);

    float4 *x, *y, *z;
    size_t size = n * sizeof(float4);
    if (UseCompressibleMemory) {
        checkCudaErrors(cudaMallocCompressible((void **)&x, size));
        checkCudaErrors(cudaMallocCompressible((void **)&y, size));
        checkCudaErrors(cudaMallocCompressible((void **)&z, size));
    }
    else {
        printf("Using non compressible memory\n");
        checkCudaErrors(cudaMalloc((void **)&x, size));
        checkCudaErrors(cudaMalloc((void **)&y, size));
        checkCudaErrors(cudaMalloc((void **)&z, size));
    }

    printf("Running saxpy on %lu bytes\n", size);

    cudaEvent_t start, stop;
    float ms;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    dim3 threads(1024, 1, 1);
    dim3 blocks;

    init<<<n / 1024, 1024>>>(x, y, z, 1.0f, n);
    checkCudaErrors(cudaDeviceSynchronize());

    // Running with single element per thread, lots of blocks
    blocks = dim3(n / threads.x, 1, 1);
    checkCudaErrors(cudaEventRecord(start));
    saxpy<<<blocks, threads>>>(1.0f, x, y, z, n);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
    printf("Running saxpy with %d blocks x %d threads = %.3f ms %.3f TB/s\n", blocks.x, threads.x, ms, (size*3)/ms/1e9);
 
    if (UseCompressibleMemory) {
        checkCudaErrors(cudaFreeCompressible(x, size));
        checkCudaErrors(cudaFreeCompressible(y, size));
        checkCudaErrors(cudaFreeCompressible(z, size));
    }
    else {
        checkCudaErrors(cudaFree(x));
        checkCudaErrors(cudaFree(y));
        checkCudaErrors(cudaFree(z));
    }

    return EXIT_SUCCESS;
}