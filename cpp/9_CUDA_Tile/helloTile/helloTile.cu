/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
 * This CUDA Tile C++ sample demonstrates basic usage of tile
 * kernels. This code launches a tile kernel using the triple chevron
 * syntax and passes data between SIMT and Tile code through global
 * device memory.  Error checks are performed using `cudaGetLastError`
 * to catch kernel launch issues and `cudaDeviceSynchronize` to catch
 * kernel execution issues.
 */

#include "helper_cuda.h"

__global__ void simtKernel(int* x) {
  printf("Hello, SIMT!\n");
  printf("[SIMT] *x == %i\n", *x);

  *x = 100;
  printf("[SIMT] *x = %i\n\n", *x);
}

__tile_global__ void tileKernel(int* x) {
  printf("Hello, Tile!\n");
  printf("[Tile] *x == %i\n", *x);

  *x = 200;
  printf("[Tile] *x = %i\n\n", *x);
}

int main() {
  int* d_x = nullptr;

  checkCudaErrors(cudaMalloc(&d_x, sizeof(int)));
  checkCudaErrors(cudaMemset(d_x, 0, sizeof(int)));

  simtKernel<<<1, 1>>>(d_x);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  /* launches tile kernel, the threads per block parameter is omitted because it must always be 1. */
  tileKernel<<<1>>>(d_x);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  int h_x = 0;
  checkCudaErrors(cudaMemcpy(&h_x, d_x, sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_x));

  printf("Hello, Host!\n");
  printf("[Host] *x == %i\n", h_x);
}
