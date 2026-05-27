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
 * This sample demonstrates a simple vector addition using CUDA Tile C++.
 * The vector addition is performed by splitting the dataset into blocks
 * which process 1024 elements at a time. The cuda::tiles::partition_view
 * type is used to partition the data into chunks of size 1024. Each block loads
 * its respective chunk from 'a' and 'b', performs an elementwise addition,
 * then stores it to the corresponding chunk of 'c'. Masked loads and stores
 * are used to ensure that the last chunk which is partially out of bounds is
 * correctly handled.
 *
 * A SIMT kernel is used to initialize the input vectors.
 */

#include "helper_cuda.h"

#include "cuda_tile.h"
#include "cuda_fp16.h"

#include <cstdio>

__global__ void initializeVectors(__half* a, __half* b, std::size_t n) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    a[idx] = __half{0.5 * idx};
    b[idx] = __half{1.5 * idx};
  }
}

/* Declares a tile kernel with '__restrict__' pointers (important for performance) */
__tile_global__ void vectorAdd(__half* __restrict__ a,
                               __half* __restrict__ b,
                               __half* __restrict__ c,
                               std::size_t n) {

  /* set up the namespace */
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  /* indicate to the compiler that the pointers are aligned (important for optimizations) */
  a = ct::assume_aligned(a, 16_ic);
  b = ct::assume_aligned(b, 16_ic);
  c = ct::assume_aligned(c, 16_ic);

  /* get the block index in the x dimension */
  auto idx = ct::bid().x;

  /* create tensor spans representing arrays of length 'n' based on the points 'a', 'b', and 'c' */
  ct::tensor_span a_span{a, ct::extents{n}};
  ct::tensor_span b_span{b, ct::extents{n}};
  ct::tensor_span c_span{c, ct::extents{n}};

  /* create partition views over the full arrays, partitioned into chunks of 1024 */
  auto view_a = ct::partition_view{a_span, ct::shape{1024_ic}};
  auto view_b = ct::partition_view{b_span, ct::shape{1024_ic}};
  auto view_c = ct::partition_view{c_span, ct::shape{1024_ic}};

  /* load the tiles from the input partitions */
  auto tile_a = view_a.load_masked(idx);
  auto tile_b = view_b.load_masked(idx);

  /* add the tiles together, elementwise */
  auto tile_c = tile_a + tile_b;

  /* store the result tile to the output partition */
  view_c.store_masked(tile_c, idx);
}

int main() {
  __half* d_a = nullptr;
  __half* d_b = nullptr;
  __half* d_c = nullptr;

  int N = 8000;
  int chunk_size = 1024;
  int num_blocks = 1 + ((N - 1) / chunk_size);

  checkCudaErrors(cudaMalloc(&d_a, N * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_b, N * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_c, N * sizeof(__half)));

  initializeVectors<<<num_blocks, chunk_size>>>(d_a, d_b, N);
  checkCudaErrors(cudaGetLastError());

  vectorAdd<<<num_blocks>>>(d_a, d_b, d_c, N);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaDeviceSynchronize());

  __half* h_c = new __half[N];
  checkCudaErrors(cudaMemcpy(h_c, d_c, N * sizeof(__half), cudaMemcpyDeviceToHost));

  for (int idx = 0; idx != N; ++idx) {
    if (h_c[idx] != __half{2 * idx}) {
      printf("Expected: h_c[%i] == %i\n", idx, 2 * idx);
      printf("Actual:   h_c[%i] == %f\n", idx, float(h_c[idx]));

      return 1;
    }
  }

  printf("Success! Vector addition matches expected results.\n");

  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_c));

  delete[] h_c;
}
