/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>

#include "scan_common.h"

int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    findCudaDevice(argc, (const char **)argv);

    uint *d_Input, *d_Output;
    uint *h_Input, *h_OutputCPU, *h_OutputGPU;
    StopWatchInterface  *hTimer = NULL;
    const uint N = 13 * 1048576 / 2;

    printf("Allocating and initializing host arrays...\n");
    sdkCreateTimer(&hTimer);
    h_Input     = (uint *)malloc(N * sizeof(uint));
    h_OutputCPU = (uint *)malloc(N * sizeof(uint));
    h_OutputGPU = (uint *)malloc(N * sizeof(uint));
    srand(2009);

    for (uint i = 0; i < N; i++)
    {
        h_Input[i] = rand();
    }

    printf("Allocating and initializing CUDA arrays...\n");
    checkCudaErrors(cudaMalloc((void **)&d_Input, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(d_Input, h_Input, N * sizeof(uint), cudaMemcpyHostToDevice));

    printf("Initializing CUDA-C scan...\n\n");
    initScan();

    int globalFlag = 1;
    size_t szWorkgroup;
    const int iCycles = 100;
    printf("*** Running GPU scan for short arrays (%d identical iterations)...\n\n", iCycles);

    for (uint arrayLength = MIN_SHORT_ARRAY_SIZE; arrayLength <= MAX_SHORT_ARRAY_SIZE; arrayLength <<= 1)
    {
        printf("Running scan for %u elements (%u arrays)...\n", arrayLength, N / arrayLength);
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);

        for (int i = 0; i < iCycles; i++)
        {
            szWorkgroup = scanExclusiveShort(d_Output, d_Input, N / arrayLength, arrayLength);
        }

        checkCudaErrors(cudaDeviceSynchronize());
        sdkStopTimer(&hTimer);
        double timerValue = 1.0e-3 * sdkGetTimerValue(&hTimer) / iCycles;

        printf("Validating the results...\n");
        printf("...reading back GPU results\n");
        checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint), cudaMemcpyDeviceToHost));

        printf(" ...scanExclusiveHost()\n");
        scanExclusiveHost(h_OutputCPU, h_Input, N / arrayLength, arrayLength);

        // Compare GPU results with CPU results and accumulate error for this test
        printf(" ...comparing the results\n");
        int localFlag = 1;

        for (uint i = 0; i < N; i++)
        {
            if (h_OutputCPU[i] != h_OutputGPU[i])
            {
                localFlag = 0;
                break;
            }
        }

        // Log message on individual test result, then accumulate to global flag
        printf(" ...Results %s\n\n", (localFlag == 1) ? "Match" : "DON'T Match !!!");
        globalFlag = globalFlag && localFlag;

        // Data log
        if (arrayLength == MAX_SHORT_ARRAY_SIZE)
        {
            printf("\n");
            printf("scan, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n",
                   (1.0e-6 * (double)arrayLength/timerValue), timerValue, (unsigned int)arrayLength, 1, (unsigned int)szWorkgroup);
            printf("\n");
        }
    }

    printf("***Running GPU scan for large arrays (%u identical iterations)...\n\n", iCycles);

    for (uint arrayLength = MIN_LARGE_ARRAY_SIZE; arrayLength <= MAX_LARGE_ARRAY_SIZE; arrayLength <<= 1)
    {
        printf("Running scan for %u elements (%u arrays)...\n", arrayLength, N / arrayLength);
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);

        for (int i = 0; i < iCycles; i++)
        {
            szWorkgroup = scanExclusiveLarge(d_Output, d_Input, N / arrayLength, arrayLength);
        }

        checkCudaErrors(cudaDeviceSynchronize());
        sdkStopTimer(&hTimer);
        double timerValue = 1.0e-3 * sdkGetTimerValue(&hTimer) / iCycles;

        printf("Validating the results...\n");
        printf("...reading back GPU results\n");
        checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint), cudaMemcpyDeviceToHost));

        printf("...scanExclusiveHost()\n");
        scanExclusiveHost(h_OutputCPU, h_Input, N / arrayLength, arrayLength);

        // Compare GPU results with CPU results and accumulate error for this test
        printf(" ...comparing the results\n");
        int localFlag = 1;

        for (uint i = 0; i < N; i++)
        {
            if (h_OutputCPU[i] != h_OutputGPU[i])
            {
                localFlag = 0;
                break;
            }
        }

        // Log message on individual test result, then accumulate to global flag
        printf(" ...Results %s\n\n", (localFlag == 1) ? "Match" : "DON'T Match !!!");
        globalFlag = globalFlag && localFlag;

        // Data log
        if (arrayLength == MAX_LARGE_ARRAY_SIZE)
        {
            printf("\n");
            printf("scan, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n",
                   (1.0e-6 * (double)arrayLength/timerValue), timerValue, (unsigned int)arrayLength, 1, (unsigned int)szWorkgroup);
            printf("\n");
        }
    }


    printf("Shutting down...\n");
    closeScan();
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));

    sdkDeleteTimer(&hTimer);

    // pass or fail (cumulative... all tests in the loop)
    exit(globalFlag ? EXIT_SUCCESS : EXIT_FAILURE);
}
