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

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication using the CUDA driver API.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 *
 * Volkov, V. 2010. Better performance at lower occupancy,
 * GPU Technology Conference 2~010 (GTC 2010).
 *
 */

// includes, system
#include <builtin_types.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstring>

// includes, project, CUDA
#include <cuda.h>
#include <helper_cuda_drvapi.h>
#include <helper_image.h>
#include <helper_string.h>
#include <helper_timer.h>

#include <cstring>
#include <iostream>
#include <string>
#include "matrixMul.h"


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);
void randomInit(float *, int);

extern "C" void computeGold(float *, const float *, const float *, unsigned int,
                            unsigned int, unsigned int);

static int initCUDA(int argc, char **argv, CUfunction *pMatrixMul,
                    int *blk_size);

#ifndef FATBIN_FILE
#define FATBIN_FILE "matrixMul_kernel64.fatbin"
#endif

////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
size_t totalGlobalMem;

const char *sSDKsample = "matrixMulDrv (Driver API)";

void constantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("[ %s ]\n", sSDKsample);

  runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  // initialize CUDA
  CUfunction matrixMul = NULL;
  int block_size = 0;

  initCUDA(argc, argv, &matrixMul, &block_size);

  // set seed for rand()
  srand(2006);

  // allocate host memory for matrices A and B
  unsigned int size_A = WA * HA;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A = reinterpret_cast<float *>(malloc(mem_size_A));
  unsigned int size_B = WB * HB;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B = reinterpret_cast<float *>(malloc(mem_size_B));

  // initialize host memory
  const float valB = 0.01f;
  constantInit(h_A, size_A, 1.0f);
  constantInit(h_B, size_B, valB);

  // allocate device memory
  CUdeviceptr d_A;
  checkCudaErrors(cuMemAlloc(&d_A, mem_size_A));
  CUdeviceptr d_B;
  checkCudaErrors(cuMemAlloc(&d_B, mem_size_B));

  // copy host memory to device
  checkCudaErrors(cuMemcpyHtoD(d_A, h_A, mem_size_A));
  checkCudaErrors(cuMemcpyHtoD(d_B, h_B, mem_size_B));

  // allocate device memory for result
  size_t size_C = WC * HC;
  size_t mem_size_C = sizeof(float) * size_C;

  CUdeviceptr d_C;
  checkCudaErrors(cuMemAlloc(&d_C, mem_size_C));

  // allocate mem for the result on host side
  float *h_C = reinterpret_cast<float *>(malloc(mem_size_C));

  // create and start timer
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);

  // start the timer
  sdkStartTimer(&timer);

  // There are two ways to launch CUDA kernels via the Driver API.
  // In this CUDA Sample, we illustrate both ways to pass parameters
  // and specify parameters.  By default we use the simpler method.
  dim3 block(block_size, block_size, 1);
  dim3 grid(WC / block_size, HC / block_size, 1);

  if (1) {
    // This is the new CUDA 4.0 API for Kernel Parameter passing and Kernel
    // Launching (simplier method)
    size_t Matrix_Width_A = (size_t)WA;
    size_t Matrix_Width_B = (size_t)WB;
    void *args[5] = {&d_C, &d_A, &d_B, &Matrix_Width_A, &Matrix_Width_B};
    // new CUDA 4.0 Driver API Kernel launch call
    checkCudaErrors(cuLaunchKernel(
        matrixMul, grid.x, grid.y, grid.z, block.x, block.y, block.z,
        2 * block_size * block_size * sizeof(float), NULL, args, NULL));
  } else {
    // This is the new CUDA 4.0 API for Kernel Parameter passing and Kernel
    // Launching (advanced method)
    int offset = 0;
    char argBuffer[256];

    // pass in launch parameters (not actually de-referencing CUdeviceptr).
    // CUdeviceptr is storing the value of the parameters
    *(reinterpret_cast<CUdeviceptr *>(&argBuffer[offset])) = d_C;
    offset += sizeof(d_C);
    *(reinterpret_cast<CUdeviceptr *>(&argBuffer[offset])) = d_A;
    offset += sizeof(d_A);
    *(reinterpret_cast<CUdeviceptr *>(&argBuffer[offset])) = d_B;
    offset += sizeof(d_B);

    size_t Matrix_Width_A = (size_t)WA;
    size_t Matrix_Width_B = (size_t)WB;

    *(reinterpret_cast<CUdeviceptr *>(&argBuffer[offset])) = Matrix_Width_A;
    offset += sizeof(Matrix_Width_A);
    *(reinterpret_cast<CUdeviceptr *>(&argBuffer[offset])) = Matrix_Width_B;
    offset += sizeof(Matrix_Width_B);

    void *kernel_launch_config[5] = {CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
                                     CU_LAUNCH_PARAM_BUFFER_SIZE, &offset,
                                     CU_LAUNCH_PARAM_END};

    // new CUDA 4.0 Driver API Kernel launch call
    checkCudaErrors(cuLaunchKernel(
        matrixMul, grid.x, grid.y, grid.z, block.x, block.y, block.z,
        2 * block_size * block_size * sizeof(float), NULL, NULL,
        reinterpret_cast<void **>(&kernel_launch_config)));
  }

  // copy result from device to host
  checkCudaErrors(cuMemcpyDtoH(reinterpret_cast<void *>(h_C), d_C, mem_size_C));

  // stop and destroy timer
  sdkStopTimer(&timer);
  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  sdkDeleteTimer(&timer);

  printf("Checking computed result for correctness: ");
  bool correct = true;

  for (int i = 0; i < static_cast<int>(WC * HC); i++) {
    if (fabs(h_C[i] - (WA * valB)) > 1e-5) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > 1e-5\n", i,
             h_C[i], WA * valB);
      correct = false;
    }
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  printf("\nNOTE: The CUDA Samples are not meant for performance measurements. "
         "Results may vary when GPU Boost is enabled.\n");

  // clean up memory
  free(h_A);
  free(h_B);
  free(h_C);
  checkCudaErrors(cuMemFree(d_A));
  checkCudaErrors(cuMemFree(d_B));
  checkCudaErrors(cuMemFree(d_C));
  checkCudaErrors(cuCtxDestroy(cuContext));
}

// Allocates a matrix with random float entries.
void randomInit(float *data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = rand() / static_cast<float>(RAND_MAX);
  }
}

static int initCUDA(int argc, char **argv, CUfunction *pMatrixMul,
                    int *blk_size) {
  CUfunction cuFunction = 0;
  int major = 0, minor = 0;
  char deviceName[100];

  cuDevice = findCudaDeviceDRV(argc, (const char **)argv);

  // get compute capabilities and the devicename
  checkCudaErrors(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  checkCudaErrors(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
  checkCudaErrors(cuDeviceGetName(deviceName, sizeof(deviceName), cuDevice));
  printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

  checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, cuDevice));
  printf("  Total amount of global memory:     %llu bytes\n",
         (long long unsigned int)totalGlobalMem);

  checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

  // first search for the module path before we load the results
  std::string module_path;
  std::ostringstream fatbin;

  if (!findFatbinPath(FATBIN_FILE, module_path, argv, fatbin)) {
    exit(EXIT_FAILURE);
  } else {
    printf("> initCUDA loading module: <%s>\n", module_path.c_str());
  }

  if (!fatbin.str().size()) {
    printf("fatbin file empty. exiting..\n");
    exit(EXIT_FAILURE);
  }

  // Create module from binary file (FATBIN)
  checkCudaErrors(cuModuleLoadData(&cuModule, fatbin.str().c_str()));

  // select the suitable kernel function
  const char *kernels[] = {"matrixMul_bs32_64bit", "matrixMul_bs16_64bit",
                           "matrixMul_bs8_64bit"};

  int idx = 0;
  int block_size = 32;
  while (idx < 3) {
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;

    checkCudaErrors(cuModuleGetFunction(&cuFunction, cuModule, kernels[idx]));
    checkCudaErrors(cuOccupancyMaxPotentialBlockSize(
        &blocksPerGrid, &threadsPerBlock, cuFunction, 0,
        2 * block_size * block_size * sizeof(float), 0));
    if (block_size * block_size <= threadsPerBlock) {
      printf("> %d block size selected\n", block_size);
      break;
    } else {
      block_size /= 2;
    }
    idx++;
  }

  *pMatrixMul = cuFunction;
  *blk_size = block_size;

  return 0;
}
