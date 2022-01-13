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

#include <helper_cuda.h>
#include <helper_string.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// Variable on the GPU used to generate unique identifiers of blocks.
////////////////////////////////////////////////////////////////////////////////
__device__ int g_uids = 0;

////////////////////////////////////////////////////////////////////////////////
// Print a simple message to signal the block which is currently executing.
////////////////////////////////////////////////////////////////////////////////
__device__ void print_info(int depth, int thread, int uid, int parent_uid) {
  if (threadIdx.x == 0) {
    if (depth == 0)
      printf("BLOCK %d launched by the host\n", uid);
    else {
      char buffer[32];

      for (int i = 0; i < depth; ++i) {
        buffer[3 * i + 0] = '|';
        buffer[3 * i + 1] = ' ';
        buffer[3 * i + 2] = ' ';
      }

      buffer[3 * depth] = '\0';
      printf("%sBLOCK %d launched by thread %d of block %d\n", buffer, uid,
             thread, parent_uid);
    }
  }

  __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// The kernel using CUDA dynamic parallelism.
//
// It generates a unique identifier for each block. Prints the information
// about that block. Finally, if the 'max_depth' has not been reached, the
// block launches new blocks directly from the GPU.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_kernel(int max_depth, int depth, int thread,
                           int parent_uid) {
  // We create a unique ID per block. Thread 0 does that and shares the value
  // with the other threads.
  __shared__ int s_uid;

  if (threadIdx.x == 0) {
    s_uid = atomicAdd(&g_uids, 1);
  }

  __syncthreads();

  // We print the ID of the block and information about its parent.
  print_info(depth, thread, s_uid, parent_uid);

  // We launch new blocks if we haven't reached the max_depth yet.
  if (++depth >= max_depth) {
    return;
  }

  cdp_kernel<<<gridDim.x, blockDim.x>>>(max_depth, depth, threadIdx.x, s_uid);
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("starting Simple Print (CUDA Dynamic Parallelism)\n");

  // Parse a few command-line arguments.
  int max_depth = 2;

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "h")) {
    printf(
        "Usage: %s depth=<max_depth>\t(where max_depth is a value between 1 "
        "and 8).\n",
        argv[0]);
    exit(EXIT_SUCCESS);
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "depth")) {
    max_depth = getCmdLineArgumentInt(argc, (const char **)argv, "depth");

    if (max_depth < 1 || max_depth > 8) {
      printf("depth parameter has to be between 1 and 8\n");
      exit(EXIT_FAILURE);
    }
  }

  // Find/set the device.
  int device = -1;
  cudaDeviceProp deviceProp;
  device = findCudaDevice(argc, (const char **)argv);
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));

  if (!(deviceProp.major > 3 ||
        (deviceProp.major == 3 && deviceProp.minor >= 5))) {
    printf("GPU %d - %s  does not support CUDA Dynamic Parallelism\n Exiting.",
           device, deviceProp.name);
    exit(EXIT_WAIVED);
  }

  // Print a message describing what the sample does.
  printf(
      "*********************************************************************"
      "******\n");
  printf(
      "The CPU launches 2 blocks of 2 threads each. On the device each thread "
      "will\n");
  printf(
      "launch 2 blocks of 2 threads each. The GPU we will do that "
      "recursively\n");
  printf("until it reaches max_depth=%d\n\n", max_depth);
  printf("In total 2");
  int num_blocks = 2, sum = 2;

  for (int i = 1; i < max_depth; ++i) {
    num_blocks *= 4;
    printf("+%d", num_blocks);
    sum += num_blocks;
  }

  printf("=%d blocks are launched!!! (%d from the GPU)\n", sum, sum - 2);
  printf(
      "************************************************************************"
      "***\n\n");

  // We set the recursion limit for CDP to max_depth.
  cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);

  // Launch the kernel from the CPU.
  printf("Launching cdp_kernel() with CUDA Dynamic Parallelism:\n\n");
  cdp_kernel<<<2, 2>>>(max_depth, 0, 0, -1);
  checkCudaErrors(cudaGetLastError());

  // Finalize.
  checkCudaErrors(cudaDeviceSynchronize());

  exit(EXIT_SUCCESS);
}
