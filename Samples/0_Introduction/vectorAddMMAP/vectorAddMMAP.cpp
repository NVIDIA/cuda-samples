/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

/* Vector addition: C = A + B.
 *
 * This sample replaces the device allocation in the vectorAddDrvsample with
 * cuMemMap-ed allocations.  This sample demonstrates that the cuMemMap api
 * allows the user to specify the physical properties of their memory while
 * retaining the contiguos nature of their access, thus not requiring a change
 * in their program structure.
 *
 */

// Includes
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <cstring>
#include <iostream>

// includes, project
#include <helper_cuda_drvapi.h>
#include <helper_functions.h>

// includes, CUDA
#include <builtin_types.h>

#include "multidevicealloc_memmap.hpp"

using namespace std;

// Variables
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction vecAdd_kernel;
float *h_A;
float *h_B;
float *h_C;
CUdeviceptr d_A;
CUdeviceptr d_B;
CUdeviceptr d_C;
size_t allocationSize = 0;

// Functions
int CleanupNoFailure();
void RandomInit(float *, int);

//define input fatbin file
#ifndef FATBIN_FILE
#define FATBIN_FILE "vectorAdd_kernel64.fatbin"
#endif

// collect all of the devices whose memory can be mapped from cuDevice.
vector<CUdevice> getBackingDevices(CUdevice cuDevice) {
  int num_devices;

  checkCudaErrors(cuDeviceGetCount(&num_devices));

  vector<CUdevice> backingDevices;
  backingDevices.push_back(cuDevice);
  for (int dev = 0; dev < num_devices; dev++) {
    int capable = 0;
    int attributeVal = 0;

    // The mapping device is already in the backingDevices vector
    if (dev == cuDevice) {
      continue;
    }

    // Only peer capable devices can map each others memory
    checkCudaErrors(cuDeviceCanAccessPeer(&capable, cuDevice, dev));
    if (!capable) {
      continue;
    }

    // The device needs to support virtual address management for the required
    // apis to work
    checkCudaErrors(cuDeviceGetAttribute(
        &attributeVal, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
        cuDevice));
    if (attributeVal == 0) {
      continue;
    }

    backingDevices.push_back(dev);
  }
  return backingDevices;
}

// Host code
int main(int argc, char **argv) {
  printf("Vector Addition (Driver API)\n");
  int N = 50000;
  size_t size = N * sizeof(float);
  int attributeVal = 0;

  // Initialize
  checkCudaErrors(cuInit(0));

  cuDevice = findCudaDeviceDRV(argc, (const char **)argv);

  // Check that the selected device supports virtual address management
  checkCudaErrors(cuDeviceGetAttribute(
      &attributeVal, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
      cuDevice));
  printf("Device %d VIRTUAL ADDRESS MANAGEMENT SUPPORTED = %d.\n", cuDevice,
         attributeVal);
  if (attributeVal == 0) {
    printf("Device %d doesn't support VIRTUAL ADDRESS MANAGEMENT.\n", cuDevice);
    exit(EXIT_WAIVED);
  }

  // The vector addition happens on cuDevice, so the allocations need to be
  // mapped there.
  vector<CUdevice> mappingDevices;
  mappingDevices.push_back(cuDevice);

  // Collect devices accessible by the mapping device (cuDevice) into the
  // backingDevices vector.
  vector<CUdevice> backingDevices = getBackingDevices(cuDevice);

  // Create context
  checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

    // first search for the module path before we load the results
    string module_path;

    std::ostringstream fatbin;

    if (!findFatbinPath(FATBIN_FILE, module_path, argv, fatbin))
    {
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("> initCUDA loading module: <%s>\n", module_path.c_str());
    }

    if (!fatbin.str().size())
    {
        printf("fatbin file empty. exiting..\n");
        exit(EXIT_FAILURE);
    }

    // Create module from binary file (FATBIN)
    checkCudaErrors(cuModuleLoadData(&cuModule, fatbin.str().c_str()));

    // Get function handle from module
    checkCudaErrors(cuModuleGetFunction(&vecAdd_kernel, cuModule, "VecAdd_kernel"));

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);


    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // Allocate vectors in device memory
    // note that a call to cuCtxEnablePeerAccess is not needed even though
    // the backing devices and mapping device are not the same.
    // This is because the cuMemSetAccess call explicitly specifies
    // the cross device mapping.
    // cuMemSetAccess is still subject to the constraints of cuDeviceCanAccessPeer
    // for cross device mappings (hence why we checked cuDeviceCanAccessPeer earlier).
    checkCudaErrors(simpleMallocMultiDeviceMmap(&d_A, &allocationSize, size, backingDevices, mappingDevices));
    checkCudaErrors(simpleMallocMultiDeviceMmap(&d_B, NULL, size, backingDevices, mappingDevices));
    checkCudaErrors(simpleMallocMultiDeviceMmap(&d_C, NULL, size, backingDevices, mappingDevices));

    // Copy vectors from host memory to device memory
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, size));
    checkCudaErrors(cuMemcpyHtoD(d_B, h_B, size));

    // This is the new CUDA 4.0 API for Kernel Parameter Passing and Kernel Launch (simpler method)

    // Grid/Block configuration
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    void *args[] = { &d_A, &d_B, &d_C, &N };

    // Launch the CUDA kernel
    checkCudaErrors(cuLaunchKernel(vecAdd_kernel,  blocksPerGrid, 1, 1,
                               threadsPerBlock, 1, 1,
                               0,
                               NULL, args, NULL));

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    checkCudaErrors(cuMemcpyDtoH(h_C, d_C, size));

    // Verify result
    int i;

    for (i = 0; i < N; ++i)
    {
        float sum = h_A[i] + h_B[i];

        if (fabs(h_C[i] - sum) > 1e-7f)
        {
            break;
        }
    }

    CleanupNoFailure();
    printf("%s\n", (i==N) ? "Result = PASS" : "Result = FAIL");

    exit((i==N) ? EXIT_SUCCESS : EXIT_FAILURE);
}

int CleanupNoFailure()
{
    // Free device memory
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_A, allocationSize));
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_B, allocationSize));
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_C, allocationSize));

    // Free host memory
    if (h_A)
    {
        free(h_A);
    }

    if (h_B)
    {
        free(h_B);
    }

    if (h_C)
    {
        free(h_C);
    }

    checkCudaErrors(cuCtxDestroy(cuContext));

    return EXIT_SUCCESS;
}
// Allocates an array with random float entries.
void RandomInit(float *data, int n)
{
    for (int i = 0; i < n; ++i)
    {
        data[i] = rand() / (float)RAND_MAX;
    }
}
