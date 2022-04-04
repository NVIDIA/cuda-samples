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

/* This sample queries the properties of the CUDA devices present
 * in the system.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda.h>
#include <helper_cuda_drvapi.h>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  CUdevice dev;
  int major = 0, minor = 0;
  int deviceCount = 0;
  char deviceName[256];

  printf("%s Starting...\n\n", argv[0]);

  // note your project will need to link with cuda.lib files on windows
  printf("CUDA Device Query (Driver API) statically linked version \n");

  checkCudaErrors(cuInit(0));

  checkCudaErrors(cuDeviceGetCount(&deviceCount));

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  for (dev = 0; dev < deviceCount; ++dev) {
    checkCudaErrors(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    checkCudaErrors(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));

    checkCudaErrors(cuDeviceGetName(deviceName, 256, dev));

    printf("\nDevice %d: \"%s\"\n", dev, deviceName);

    int driverVersion = 0;
    checkCudaErrors(cuDriverGetVersion(&driverVersion));
    printf("  CUDA Driver Version:                           %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n", major,
           minor);

    size_t totalGlobalMem;
    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, dev));

    char msg[256];
    SPRINTF(msg,
            "  Total amount of global memory:                 %.0f MBytes "
            "(%llu bytes)\n",
            (float)totalGlobalMem / 1048576.0f,
            (unsigned long long)totalGlobalMem);
    printf("%s", msg);

    int multiProcessorCount;
    getCudaAttribute<int>(&multiProcessorCount,
                          CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);

    printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
           multiProcessorCount, _ConvertSMVer2CoresDRV(major, minor),
           _ConvertSMVer2CoresDRV(major, minor) * multiProcessorCount);

    int clockRate;
    getCudaAttribute<int>(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
    printf(
        "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
        "GHz)\n",
        clockRate * 1e-3f, clockRate * 1e-6f);
    int memoryClock;
    getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                          dev);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           memoryClock * 1e-3f);
    int memBusWidth;
    getCudaAttribute<int>(&memBusWidth,
                          CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
    printf("  Memory Bus Width:                              %d-bit\n",
           memBusWidth);
    int L2CacheSize;
    getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

    if (L2CacheSize) {
      printf("  L2 Cache Size:                                 %d bytes\n",
             L2CacheSize);
    }

    int maxTex1D, maxTex2D[2], maxTex3D[3];
    getCudaAttribute<int>(&maxTex1D,
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, dev);
    getCudaAttribute<int>(&maxTex2D[0],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, dev);
    getCudaAttribute<int>(&maxTex2D[1],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, dev);
    getCudaAttribute<int>(&maxTex3D[0],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, dev);
    getCudaAttribute<int>(&maxTex3D[1],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, dev);
    getCudaAttribute<int>(&maxTex3D[2],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, dev);
    printf(
        "  Max Texture Dimension Sizes                    1D=(%d) 2D=(%d, %d) "
        "3D=(%d, %d, %d)\n",
        maxTex1D, maxTex2D[0], maxTex2D[1], maxTex3D[0], maxTex3D[1],
        maxTex3D[2]);

    int maxTex1DLayered[2];
    getCudaAttribute<int>(&maxTex1DLayered[0],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
                          dev);
    getCudaAttribute<int>(&maxTex1DLayered[1],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
                          dev);
    printf(
        "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
        maxTex1DLayered[0], maxTex1DLayered[1]);

    int maxTex2DLayered[3];
    getCudaAttribute<int>(&maxTex2DLayered[0],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
                          dev);
    getCudaAttribute<int>(&maxTex2DLayered[1],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
                          dev);
    getCudaAttribute<int>(&maxTex2DLayered[2],
                          CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
                          dev);
    printf(
        "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
        "layers\n",
        maxTex2DLayered[0], maxTex2DLayered[1], maxTex2DLayered[2]);

    int totalConstantMemory;
    getCudaAttribute<int>(&totalConstantMemory,
                          CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, dev);
    printf("  Total amount of constant memory:               %u bytes\n",
           totalConstantMemory);
    int sharedMemPerBlock;
    getCudaAttribute<int>(&sharedMemPerBlock,
                          CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev);
    printf("  Total amount of shared memory per block:       %u bytes\n",
           sharedMemPerBlock);
    int regsPerBlock;
    getCudaAttribute<int>(&regsPerBlock,
                          CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev);
    printf("  Total number of registers available per block: %d\n",
           regsPerBlock);
    int warpSize;
    getCudaAttribute<int>(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);
    printf("  Warp size:                                     %d\n", warpSize);
    int maxThreadsPerMultiProcessor;
    getCudaAttribute<int>(&maxThreadsPerMultiProcessor,
                          CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                          dev);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           maxThreadsPerMultiProcessor);
    int maxThreadsPerBlock;
    getCudaAttribute<int>(&maxThreadsPerBlock,
                          CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
    printf("  Maximum number of threads per block:           %d\n",
           maxThreadsPerBlock);

    int blockDim[3];
    getCudaAttribute<int>(&blockDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                          dev);
    getCudaAttribute<int>(&blockDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                          dev);
    getCudaAttribute<int>(&blockDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                          dev);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           blockDim[0], blockDim[1], blockDim[2]);
    int gridDim[3];
    getCudaAttribute<int>(&gridDim[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev);
    getCudaAttribute<int>(&gridDim[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev);
    getCudaAttribute<int>(&gridDim[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev);
    printf("  Max dimension size of a grid size (x,y,z):    (%d, %d, %d)\n",
           gridDim[0], gridDim[1], gridDim[2]);

    int textureAlign;
    getCudaAttribute<int>(&textureAlign, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
                          dev);
    printf("  Texture alignment:                             %u bytes\n",
           textureAlign);

    int memPitch;
    getCudaAttribute<int>(&memPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, dev);
    printf("  Maximum memory pitch:                          %u bytes\n",
           memPitch);

    int gpuOverlap;
    getCudaAttribute<int>(&gpuOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, dev);

    int asyncEngineCount;
    getCudaAttribute<int>(&asyncEngineCount,
                          CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, dev);
    printf(
        "  Concurrent copy and kernel execution:          %s with %d copy "
        "engine(s)\n",
        (gpuOverlap ? "Yes" : "No"), asyncEngineCount);

    int kernelExecTimeoutEnabled;
    getCudaAttribute<int>(&kernelExecTimeoutEnabled,
                          CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, dev);
    printf("  Run time limit on kernels:                     %s\n",
           kernelExecTimeoutEnabled ? "Yes" : "No");
    int integrated;
    getCudaAttribute<int>(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev);
    printf("  Integrated GPU sharing Host Memory:            %s\n",
           integrated ? "Yes" : "No");
    int canMapHostMemory;
    getCudaAttribute<int>(&canMapHostMemory,
                          CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev);
    printf("  Support host page-locked memory mapping:       %s\n",
           canMapHostMemory ? "Yes" : "No");

    int concurrentKernels;
    getCudaAttribute<int>(&concurrentKernels,
                          CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, dev);
    printf("  Concurrent kernel execution:                   %s\n",
           concurrentKernels ? "Yes" : "No");

    int surfaceAlignment;
    getCudaAttribute<int>(&surfaceAlignment,
                          CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, dev);
    printf("  Alignment requirement for Surfaces:            %s\n",
           surfaceAlignment ? "Yes" : "No");

    int eccEnabled;
    getCudaAttribute<int>(&eccEnabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, dev);
    printf("  Device has ECC support:                        %s\n",
           eccEnabled ? "Enabled" : "Disabled");

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    int tccDriver;
    getCudaAttribute<int>(&tccDriver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, dev);
    printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
           tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                     : "WDDM (Windows Display Driver Model)");
#endif

    int unifiedAddressing;
    getCudaAttribute<int>(&unifiedAddressing,
                          CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
    printf("  Device supports Unified Addressing (UVA):      %s\n",
           unifiedAddressing ? "Yes" : "No");

    int managedMemory;
    getCudaAttribute<int>(&managedMemory, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
                          dev);
    printf("  Device supports Managed Memory:                %s\n",
           managedMemory ? "Yes" : "No");

    int computePreemption;
    getCudaAttribute<int>(&computePreemption,
                          CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED,
                          dev);
    printf("  Device supports Compute Preemption:            %s\n",
           computePreemption ? "Yes" : "No");

    int cooperativeLaunch;
    getCudaAttribute<int>(&cooperativeLaunch,
                          CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, dev);
    printf("  Supports Cooperative Kernel Launch:            %s\n",
           cooperativeLaunch ? "Yes" : "No");

    int cooperativeMultiDevLaunch;
    getCudaAttribute<int>(&cooperativeMultiDevLaunch,
                          CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH,
                          dev);
    printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
           cooperativeMultiDevLaunch ? "Yes" : "No");

    int pciDomainID, pciBusID, pciDeviceID;
    getCudaAttribute<int>(&pciDomainID, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev);
    getCudaAttribute<int>(&pciBusID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev);
    getCudaAttribute<int>(&pciDeviceID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev);
    printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
           pciDomainID, pciBusID, pciDeviceID);

    const char *sComputeMode[] = {
        "Default (multiple host threads can use ::cudaSetDevice() with device "
        "simultaneously)",
        "Exclusive (only one host thread in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this "
        "device)",
        "Exclusive Process (many threads in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Unknown", NULL};

    int computeMode;
    getCudaAttribute<int>(&computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev);
    printf("  Compute Mode:\n");
    printf("     < %s >\n", sComputeMode[computeMode]);
  }

  // If there are 2 or more GPUs, query to determine whether RDMA is supported
  if (deviceCount >= 2) {
    int gpuid[64];  // we want to find the first two GPUs that can support P2P
    int gpu_p2p_count = 0;
    int tccDriver = 0;

    for (int i = 0; i < deviceCount; i++) {
      checkCudaErrors(cuDeviceGetAttribute(
          &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, i));
      checkCudaErrors(cuDeviceGetAttribute(
          &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, i));
      getCudaAttribute<int>(&tccDriver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, i);

      // Only boards based on Fermi or later can support P2P
      if ((major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
          // on Windows (64-bit), the Tesla Compute Cluster driver for windows
          // must be enabled to support this
          && tccDriver
#endif
          ) {
        // This is an array of P2P capable GPUs
        gpuid[gpu_p2p_count++] = i;
      }
    }

    // Show all the combinations of support P2P GPUs
    int can_access_peer;
    char deviceName0[256], deviceName1[256];

    if (gpu_p2p_count >= 2) {
      for (int i = 0; i < gpu_p2p_count; i++) {
        for (int j = 0; j < gpu_p2p_count; j++) {
          if (gpuid[i] == gpuid[j]) {
            continue;
          }
          checkCudaErrors(
              cuDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
          checkCudaErrors(cuDeviceGetName(deviceName0, 256, gpuid[i]));
          checkCudaErrors(cuDeviceGetName(deviceName1, 256, gpuid[j]));
          printf(
              "> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : "
              "%s\n",
              deviceName0, gpuid[i], deviceName1, gpuid[j],
              can_access_peer ? "Yes" : "No");
        }
      }
    }
  }

  printf("Result = PASS\n");

  exit(EXIT_SUCCESS);
}
