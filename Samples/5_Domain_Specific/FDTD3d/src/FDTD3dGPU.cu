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

#include "FDTD3dGPU.h"

#include <iostream>
#include <algorithm>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "FDTD3dGPUKernel.cuh"

bool getTargetDeviceGlobalMemSize(memsize_t *result, const int argc,
                                  const char **argv) {
  int deviceCount = 0;
  int targetDevice = 0;
  size_t memsize = 0;

  // Get the number of CUDA enabled GPU devices
  printf(" cudaGetDeviceCount\n");
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  // Select target device (device 0 by default)
  targetDevice = findCudaDevice(argc, (const char **)argv);

  // Query target device for maximum memory allocation
  printf(" cudaGetDeviceProperties\n");
  struct cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, targetDevice));

  memsize = deviceProp.totalGlobalMem;

  // Save the result
  *result = (memsize_t)memsize;
  return true;
}

bool fdtdGPU(float *output, const float *input, const float *coeff,
             const int dimx, const int dimy, const int dimz, const int radius,
             const int timesteps, const int argc, const char **argv) {
  const int outerDimx = dimx + 2 * radius;
  const int outerDimy = dimy + 2 * radius;
  const int outerDimz = dimz + 2 * radius;
  const size_t volumeSize = outerDimx * outerDimy * outerDimz;
  int deviceCount = 0;
  int targetDevice = 0;
  float *bufferOut = 0;
  float *bufferIn = 0;
  dim3 dimBlock;
  dim3 dimGrid;

  // Ensure that the inner data starts on a 128B boundary
  const int padding = (128 / sizeof(float)) - radius;
  const size_t paddedVolumeSize = volumeSize + padding;

#ifdef GPU_PROFILING
  cudaEvent_t profileStart = 0;
  cudaEvent_t profileEnd = 0;
  const int profileTimesteps = timesteps - 1;

  if (profileTimesteps < 1) {
    printf(
        " cannot profile with fewer than two timesteps (timesteps=%d), "
        "profiling is disabled.\n",
        timesteps);
  }

#endif

  // Check the radius is valid
  if (radius != RADIUS) {
    printf("radius is invalid, must be %d - see kernel for details.\n", RADIUS);
    exit(EXIT_FAILURE);
  }

  // Get the number of CUDA enabled GPU devices
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  // Select target device (device 0 by default)
  targetDevice = findCudaDevice(argc, (const char **)argv);

  checkCudaErrors(cudaSetDevice(targetDevice));

  // Allocate memory buffers
  checkCudaErrors(
      cudaMalloc((void **)&bufferOut, paddedVolumeSize * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&bufferIn, paddedVolumeSize * sizeof(float)));

  // Check for a command-line specified block size
  int userBlockSize;

  if (checkCmdLineFlag(argc, (const char **)argv, "block-size")) {
    userBlockSize = getCmdLineArgumentInt(argc, argv, "block-size");
    // Constrain to a multiple of k_blockDimX
    userBlockSize = (userBlockSize / k_blockDimX * k_blockDimX);

    // Constrain within allowed bounds
    userBlockSize = MIN(MAX(userBlockSize, k_blockSizeMin), k_blockSizeMax);
  } else {
    userBlockSize = k_blockSizeMax;
  }

  // Check the device limit on the number of threads
  struct cudaFuncAttributes funcAttrib;
  checkCudaErrors(cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel));

  userBlockSize = MIN(userBlockSize, funcAttrib.maxThreadsPerBlock);

  // Set the block size
  dimBlock.x = k_blockDimX;
  // Visual Studio 2005 does not like std::min
  //    dimBlock.y = std::min<size_t>(userBlockSize / k_blockDimX,
  //    (size_t)k_blockDimMaxY);
  dimBlock.y = ((userBlockSize / k_blockDimX) < (size_t)k_blockDimMaxY)
                   ? (userBlockSize / k_blockDimX)
                   : (size_t)k_blockDimMaxY;
  dimGrid.x = (unsigned int)ceil((float)dimx / dimBlock.x);
  dimGrid.y = (unsigned int)ceil((float)dimy / dimBlock.y);
  printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
  printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);

  // Check the block size is valid
  if (dimBlock.x < RADIUS || dimBlock.y < RADIUS) {
    printf("invalid block size, x (%d) and y (%d) must be >= radius (%d).\n",
           dimBlock.x, dimBlock.y, RADIUS);
    exit(EXIT_FAILURE);
  }

  // Copy the input to the device input buffer
  checkCudaErrors(cudaMemcpy(bufferIn + padding, input,
                             volumeSize * sizeof(float),
                             cudaMemcpyHostToDevice));

  // Copy the input to the device output buffer (actually only need the halo)
  checkCudaErrors(cudaMemcpy(bufferOut + padding, input,
                             volumeSize * sizeof(float),
                             cudaMemcpyHostToDevice));

  // Copy the coefficients to the device coefficient buffer
  checkCudaErrors(
      cudaMemcpyToSymbol(stencil, (void *)coeff, (radius + 1) * sizeof(float)));

#ifdef GPU_PROFILING

  // Create the events
  checkCudaErrors(cudaEventCreate(&profileStart));
  checkCudaErrors(cudaEventCreate(&profileEnd));

#endif

  // Execute the FDTD
  float *bufferSrc = bufferIn + padding;
  float *bufferDst = bufferOut + padding;
  printf(" GPU FDTD loop\n");

#ifdef GPU_PROFILING
  // Enqueue start event
  checkCudaErrors(cudaEventRecord(profileStart, 0));
#endif

  for (int it = 0; it < timesteps; it++) {
    printf("\tt = %d ", it);

    // Launch the kernel
    printf("launch kernel\n");
    FiniteDifferencesKernel<<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx,
                                                   dimy, dimz);

    // Toggle the buffers
    // Visual Studio 2005 does not like std::swap
    //    std::swap<float *>(bufferSrc, bufferDst);
    float *tmp = bufferDst;
    bufferDst = bufferSrc;
    bufferSrc = tmp;
  }

  printf("\n");

#ifdef GPU_PROFILING
  // Enqueue end event
  checkCudaErrors(cudaEventRecord(profileEnd, 0));
#endif

  // Wait for the kernel to complete
  checkCudaErrors(cudaDeviceSynchronize());

  // Read the result back, result is in bufferSrc (after final toggle)
  checkCudaErrors(cudaMemcpy(output, bufferSrc, volumeSize * sizeof(float),
                             cudaMemcpyDeviceToHost));

// Report time
#ifdef GPU_PROFILING
  float elapsedTimeMS = 0;

  if (profileTimesteps > 0) {
    checkCudaErrors(
        cudaEventElapsedTime(&elapsedTimeMS, profileStart, profileEnd));
  }

  if (profileTimesteps > 0) {
    // Convert milliseconds to seconds
    double elapsedTime = elapsedTimeMS * 1.0e-3;
    double avgElapsedTime = elapsedTime / (double)profileTimesteps;
    // Determine number of computations per timestep
    size_t pointsComputed = dimx * dimy * dimz;
    // Determine throughput
    double throughputM = 1.0e-6 * (double)pointsComputed / avgElapsedTime;
    printf(
        "FDTD3d, Throughput = %.4f MPoints/s, Time = %.5f s, Size = %u Points, "
        "NumDevsUsed = %u, Blocksize = %u\n",
        throughputM, avgElapsedTime, pointsComputed, 1,
        dimBlock.x * dimBlock.y);
  }

#endif

  // Cleanup
  if (bufferIn) {
    checkCudaErrors(cudaFree(bufferIn));
  }

  if (bufferOut) {
    checkCudaErrors(cudaFree(bufferOut));
  }

#ifdef GPU_PROFILING

  if (profileStart) {
    checkCudaErrors(cudaEventDestroy(profileStart));
  }

  if (profileEnd) {
    checkCudaErrors(cudaEventDestroy(profileEnd));
  }

#endif
  return true;
}
