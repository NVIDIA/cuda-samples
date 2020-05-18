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

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

cudaAccessPolicyWindow
initAccessPolicyWindow(void) {
   cudaAccessPolicyWindow accessPolicyWindow = { 0 };
   accessPolicyWindow.base_ptr = (void *)0;
   accessPolicyWindow.num_bytes = 0;
   accessPolicyWindow.hitRatio = 0.f;
   accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
   accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
   return accessPolicyWindow;
}
////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param data  input data in global memory
//! @param dataSize  input data size
//! @param bigData  input bigData in global memory
//! @param bigDataSize  input bigData size
//! @param hitcount how many data access are done within block
////////////////////////////////////////////////////////////////////////////////
static __global__ void
kernCacheSegmentTest(int* data, int dataSize, int *trash, int bigDataSize, int hitCount)
{
    __shared__ unsigned int hit;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tID = row * blockDim.y + col;
    uint32_t psRand = tID;

    atomicExch(&hit, 0);
    __syncthreads();
    while (hit < hitCount) {
         psRand ^= psRand << 13;
         psRand ^= psRand >> 17;
         psRand ^= psRand << 5;

         int idx = tID - psRand;
         if (idx < 0) {
             idx = -idx;
         }

         if((tID % 2) == 0) {
             data[psRand % dataSize] = data[psRand % dataSize] + data[idx % dataSize];
         } else {
             trash[psRand % bigDataSize] = trash[psRand % bigDataSize] + trash[idx % bigDataSize];
         }

        atomicAdd(&hit, 1);
    }
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    bool bTestResult = true;
    cudaAccessPolicyWindow accessPolicyWindow;
    cudaDeviceProp deviceProp;
    cudaStreamAttrValue streamAttrValue;
    cudaStream_t stream;
    cudaStreamAttrID streamAttrID;
    dim3 threads(32, 32);
    int *dataDevicePointer;
    int *dataHostPointer;
    int dataSize;
    int *bigDataDevicePointer;
    int *bigDataHostPointer;
    int bigDataSize;
    StopWatchInterface *timer = 0;

    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    //Get device properties
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    dim3 blocks(deviceProp.maxGridSize[1], 1);

    //Make sure device the l2 optimization
    if (deviceProp.persistingL2CacheMaxSize == 0) {
        printf("Waiving execution as device %d does not support persisting L2 Caching\n", devID);
       exit(EXIT_WAIVED);
    }

    //Create stream to assiocate with window
    checkCudaErrors(cudaStreamCreate(&stream));

    //Set the amount of l2 cache that will be persisting to maximum the device can support
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, deviceProp.persistingL2CacheMaxSize));

    //Stream attribute to set
    streamAttrID = cudaStreamAttributeAccessPolicyWindow;

    //Default window
    streamAttrValue.accessPolicyWindow      = initAccessPolicyWindow();
    accessPolicyWindow                      = initAccessPolicyWindow();

    //Allocate size of both buffers
    bigDataSize = (deviceProp.l2CacheSize * 4) / sizeof(int);
    dataSize = (deviceProp.l2CacheSize / 4)  / sizeof(int);

    //Allocate data
    dataHostPointer = (int *)malloc(dataSize * sizeof(int));
    bigDataHostPointer = (int *)malloc(bigDataSize * sizeof(int));

    for ( int i = 0; i < bigDataSize; ++i) {
        if (i < dataSize) {
            dataHostPointer[i] = i;
        }

        bigDataHostPointer[bigDataSize - i - 1] = i;
    }

    checkCudaErrors(cudaMalloc((void**) &dataDevicePointer, dataSize * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &bigDataDevicePointer, bigDataSize * sizeof(int)));
    checkCudaErrors(cudaMemcpyAsync(dataDevicePointer, dataHostPointer, dataSize * sizeof(int), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(bigDataDevicePointer, bigDataHostPointer, bigDataSize * sizeof(int), cudaMemcpyHostToDevice, stream));

    //Make a window for the buffer of interest
    accessPolicyWindow.base_ptr = (void *)dataDevicePointer;
    accessPolicyWindow.num_bytes = dataSize * sizeof(int);
    accessPolicyWindow.hitRatio = 1.f;
    accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    streamAttrValue.accessPolicyWindow = accessPolicyWindow;

    //Assign window to stream
    checkCudaErrors(cudaStreamSetAttribute(stream, streamAttrID, &streamAttrValue));

    //Demote any previous persisting lines
    checkCudaErrors(cudaCtxResetPersistingL2Cache());

    checkCudaErrors(cudaStreamSynchronize(stream));
    kernCacheSegmentTest<<<blocks, threads, 0, stream>>>(dataDevicePointer, dataSize, bigDataDevicePointer, bigDataSize, 0xAFFFF);

    checkCudaErrors(cudaStreamSynchronize(stream));
    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    //Free memory
    free(dataHostPointer);
    free(bigDataHostPointer);
    checkCudaErrors(cudaFree(dataDevicePointer));
    checkCudaErrors(cudaFree(bigDataDevicePointer));

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
