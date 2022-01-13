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

#ifndef QUASIRANDOMGENERATOR_GPU_CUH
#define QUASIRANDOMGENERATOR_GPU_CUH

#include <nvrtc_helper.h>
#include "quasirandomGenerator_common.h"

// Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

// Global variables for nvrtc outputs
char *cubin;
size_t cubinSize;
CUmodule module;

////////////////////////////////////////////////////////////////////////////////
// GPU code
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Niederreiter quasirandom number generation kernel
////////////////////////////////////////////////////////////////////////////////

// Table initialization routine
void initTableGPU(unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION]) {
  CUdeviceptr c_Table;
  checkCudaErrors(cuModuleGetGlobal(&c_Table, NULL, module, "c_Table"));
  checkCudaErrors(
      cuMemcpyHtoD(c_Table, tableCPU,
                   QRNG_DIMENSIONS * QRNG_RESOLUTION * sizeof(unsigned int)));
}

// Host-side interface
void quasirandomGeneratorGPU(CUdeviceptr d_Output, unsigned int seed,
                             unsigned int N) {
  dim3 threads(128, QRNG_DIMENSIONS);
  dim3 cudaGridSize(128, 1, 1);

  CUfunction kernel_addr;
  checkCudaErrors(
      cuModuleGetFunction(&kernel_addr, module, "quasirandomGeneratorKernel"));

  void *args[] = {(void *)&d_Output, (void *)&seed, (void *)&N};
  checkCudaErrors(cuLaunchKernel(kernel_addr, cudaGridSize.x, cudaGridSize.y,
                                 cudaGridSize.z, /* grid dim */
                                 threads.x, threads.y,
                                 threads.z, /* block dim */
                                 0, 0,      /* shared mem, stream */
                                 &args[0],  /* arguments */
                                 0));

  checkCudaErrors(cuCtxSynchronize());
}

void inverseCNDgpu(CUdeviceptr d_Output, unsigned int N) {
  dim3 threads(128, 1, 1);
  dim3 cudaGridSize(128, 1, 1);

  CUfunction kernel_addr;
  checkCudaErrors(
      cuModuleGetFunction(&kernel_addr, module, "inverseCNDKernel"));

  void *args[] = {(void *)&d_Output, (void *)&N};
  checkCudaErrors(cuLaunchKernel(kernel_addr, cudaGridSize.x, cudaGridSize.y,
                                 cudaGridSize.z, /* grid dim */
                                 threads.x, threads.y,
                                 threads.z, /* block dim */
                                 0, 0,      /* shared mem, stream */
                                 &args[0],  /* arguments */
                                 0));

  checkCudaErrors(cuCtxSynchronize());
}

#endif
