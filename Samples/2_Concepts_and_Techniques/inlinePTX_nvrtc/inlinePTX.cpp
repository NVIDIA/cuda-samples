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

/*
 * Demonstration of inline PTX (assembly language) usage in CUDA kernels
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <nvrtc_helper.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>

void sequence_cpu(int *h_ptr, int length) {
  for (int elemID = 0; elemID < length; elemID++) {
    h_ptr[elemID] = elemID % 32;
  }
}

int main(int argc, char **argv) {
  printf("CUDA inline PTX assembler sample\n");

  char *cubin, *kernel_file;
  size_t cubinSize;

  kernel_file = sdkFindFilePath("inlinePTX_kernel.cu", argv[0]);
  compileFileToCUBIN(kernel_file, argc, argv, &cubin, &cubinSize, 0);

  CUmodule module = loadCUBIN(cubin, argc, argv);

  CUfunction kernel_addr;

  checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "sequence_gpu"));

  const int N = 1000;
  int *h_ptr = (int *)malloc(N * sizeof(int));

  dim3 cudaBlockSize(256, 1, 1);
  dim3 cudaGridSize((N + cudaBlockSize.x - 1) / cudaBlockSize.x, 1, 1);

  CUdeviceptr d_ptr;
  checkCudaErrors(cuMemAlloc(&d_ptr, N * sizeof(int)));

  void *arr[] = {(void *)&d_ptr, (void *)&N};
  checkCudaErrors(cuLaunchKernel(kernel_addr, cudaGridSize.x, cudaGridSize.y,
                                 cudaGridSize.z, /* grid dim */
                                 cudaBlockSize.x, cudaBlockSize.y,
                                 cudaBlockSize.z, /* block dim */
                                 0, 0,            /* shared mem, stream */
                                 &arr[0],         /* arguments */
                                 0));

  checkCudaErrors(cuCtxSynchronize());

  sequence_cpu(h_ptr, N);

  int *h_d_ptr = (int *)malloc(N * sizeof(int));
  checkCudaErrors(cuMemcpyDtoH(h_d_ptr, d_ptr, N * sizeof(int)));

  bool bValid = true;

  for (int i = 0; i < N && bValid; i++) {
    if (h_ptr[i] != h_d_ptr[i]) {
      bValid = false;
    }
  }

  printf("Test %s.\n", bValid ? "Successful" : "Failed");

  checkCudaErrors(cuMemFree(d_ptr));

  return bValid ? EXIT_SUCCESS : EXIT_FAILURE;
}
