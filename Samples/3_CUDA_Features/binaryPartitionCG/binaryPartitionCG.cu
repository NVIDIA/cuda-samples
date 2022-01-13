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
 * This sample illustrates basic usage of binary partition cooperative groups
 * within the thread block tile when divergent path exists.
 * 1.) Each thread loads a value from random array.
 * 2.) then checks if it is odd or even.
 * 3.) create binary partition group based on the above predicate
 * 4.) we count the number of odd/even in the group based on size of the binary
       groups
 * 5.) write it global counter of odd.
 * 6.) sum the values loaded by individual threads(using reduce) and write it to
       global even & odd elements sum.
 *
 * **NOTE** :
 *    binary_partition results in splitting warp into divergent thread groups
 *    this is not good from performance perspective, but in cases where warp
 *    divergence is inevitable one can use binary_partition group.
*/

#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <helper_cuda.h>

namespace cg = cooperative_groups;

void initOddEvenArr(int *inputArr, unsigned int size) {
  for (int i = 0; i < size; i++) {
    inputArr[i] = rand() % 50;
  }
}

/**
 * CUDA kernel device code
 *
 * Creates cooperative groups and performs odd/even counting & summation.
 */
__global__ void oddEvenCountAndSumCG(int *inputArr, int *numOfOdds,
                                     int *sumOfOddAndEvens, unsigned int size) {
  cg::thread_block cta = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    int elem = inputArr[i];
    auto subTile = cg::binary_partition(tile32, elem & 1);
    if (elem & 1)  // Odd numbers group
    {
      int oddGroupSum = cg::reduce(subTile, elem, cg::plus<int>());

      if (subTile.thread_rank() == 0) {
        // Add number of odds present in this group of Odds.
        atomicAdd(numOfOdds, subTile.size());

        // Add local reduction of odds present in this group of Odds.
        atomicAdd(&sumOfOddAndEvens[0], oddGroupSum);
      }
    } else  // Even numbers group
    {
      int evenGroupSum = cg::reduce(subTile, elem, cg::plus<int>());

      if (subTile.thread_rank() == 0) {
        // Add local reduction of even present in this group of evens.
        atomicAdd(&sumOfOddAndEvens[1], evenGroupSum);
      }
    }
    // reconverge warp so for next loop iteration we ensure convergence of
    // above diverged threads to perform coalesced loads of inputArr.
    cg::sync(tile32);
  }
}

/**
 * Host main routine
 */
int main(int argc, const char **argv) {
  int deviceId = findCudaDevice(argc, argv);
  int *h_inputArr, *d_inputArr;
  int *h_numOfOdds, *d_numOfOdds;
  int *h_sumOfOddEvenElems, *d_sumOfOddEvenElems;
  unsigned int arrSize = 1024 * 100;

  checkCudaErrors(cudaMallocHost(&h_inputArr, sizeof(int) * arrSize));
  checkCudaErrors(cudaMallocHost(&h_numOfOdds, sizeof(int)));
  checkCudaErrors(cudaMallocHost(&h_sumOfOddEvenElems, sizeof(int) * 2));
  initOddEvenArr(h_inputArr, arrSize);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  checkCudaErrors(cudaMalloc(&d_inputArr, sizeof(int) * arrSize));
  checkCudaErrors(cudaMalloc(&d_numOfOdds, sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_sumOfOddEvenElems, sizeof(int) * 2));

  checkCudaErrors(cudaMemcpyAsync(d_inputArr, h_inputArr, sizeof(int) * arrSize,
                                  cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemsetAsync(d_numOfOdds, 0, sizeof(int), stream));
  checkCudaErrors(
      cudaMemsetAsync(d_sumOfOddEvenElems, 0, 2 * sizeof(int), stream));

  // Launch the kernel
  int threadsPerBlock = 0;
  int blocksPerGrid = 0;
  checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
      &blocksPerGrid, &threadsPerBlock, oddEvenCountAndSumCG, 0, 0));

  printf("\nLaunching %d blocks with %d threads...\n\n", blocksPerGrid,
         threadsPerBlock);

  oddEvenCountAndSumCG<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
      d_inputArr, d_numOfOdds, d_sumOfOddEvenElems, arrSize);

  checkCudaErrors(cudaMemcpyAsync(h_numOfOdds, d_numOfOdds, sizeof(int),
                                  cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaMemcpyAsync(h_sumOfOddEvenElems, d_sumOfOddEvenElems,
                                  2 * sizeof(int), cudaMemcpyDeviceToHost,
                                  stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  printf("Array size = %d Num of Odds = %d Sum of Odds = %d Sum of Evens %d\n",
         arrSize, h_numOfOdds[0], h_sumOfOddEvenElems[0],
         h_sumOfOddEvenElems[1]);
  printf("\n...Done.\n\n");

  checkCudaErrors(cudaFreeHost(h_inputArr));
  checkCudaErrors(cudaFreeHost(h_numOfOdds));
  checkCudaErrors(cudaFreeHost(h_sumOfOddEvenElems));

  checkCudaErrors(cudaFree(d_inputArr));
  checkCudaErrors(cudaFree(d_numOfOdds));
  checkCudaErrors(cudaFree(d_sumOfOddEvenElems));

  return EXIT_SUCCESS;
}
