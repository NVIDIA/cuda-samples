/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <stdio.h>
// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>

#include <cuda_runtime.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define NUM_ELEMS 10000000
#define NUM_THREADS_PER_BLOCK 512

// warp-aggregated atomic increment
__device__ int atomicAggInc(int *counter) {
  cg::coalesced_group active = cg::coalesced_threads();

  // leader does the update
  int res = 0;
  if (active.thread_rank() == 0) {
    res = atomicAdd(counter, active.size());
  }

  // broadcast result
  res = active.shfl(res, 0);

  // each thread computes its own value
  return res + active.thread_rank();
}

__global__ void filter_arr(int *dst, int *nres, const int *src, int n) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  for (int i = id; i < n; i += gridDim.x * blockDim.x) {
    if (src[i] > 0) dst[atomicAggInc(nres)] = src[i];
  }
}

// warp-aggregated atomic multi bucket increment
#if __CUDA_ARCH__ >= 700
__device__ int atomicAggIncMulti(const int bucket, int *counter)
{
  cg::coalesced_group active = cg::coalesced_threads();
  // group all threads with same bucket value.
  auto labeledGroup = cg::labeled_partition(active, bucket);

  int res = 0;
  if (labeledGroup.thread_rank() == 0)
  {
    res = atomicAdd(&counter[bucket], labeledGroup.size());
  }

  // broadcast result
  res = labeledGroup.shfl(res, 0);

  // each thread computes its own value
  return res + labeledGroup.thread_rank();
}
#endif

// Places individual value indices into its corresponding buckets.
__global__ void mapToBuckets(const int *srcArr, int *indicesBuckets, int *bucketCounters, const int srcSize, const int numOfBuckets)
{
#if __CUDA_ARCH__ >= 700
  cg::grid_group grid = cg::this_grid();

  for (int i=grid.thread_rank(); i < srcSize; i += grid.size())
  {
    const int bucket = srcArr[i];
    if (bucket < numOfBuckets)
    {
      indicesBuckets[atomicAggIncMulti(bucket, bucketCounters)] = i;
    }
  }
#endif
}

int mapIndicesToBuckets(int *h_srcArr, int *d_srcArr, int numOfBuckets)
{
  int *d_indicesBuckets, *d_bucketCounters;
  int *cpuBucketCounters = new int[numOfBuckets];
  int *h_bucketCounters = new int[numOfBuckets];

  memset(cpuBucketCounters, 0, sizeof(int)*numOfBuckets);
  // Initialize each bucket counters.
  for (int i = 0; i < numOfBuckets; i++)
  {
    h_bucketCounters[i] = i*NUM_ELEMS;
  }

  checkCudaErrors(cudaMalloc(&d_indicesBuckets, sizeof(int) * NUM_ELEMS * numOfBuckets));
  checkCudaErrors(cudaMalloc(&d_bucketCounters, sizeof(int) * numOfBuckets));

  checkCudaErrors(cudaMemcpy(d_bucketCounters, h_bucketCounters, sizeof(int)*numOfBuckets, cudaMemcpyHostToDevice));

  dim3 dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
  dim3 dimGrid((NUM_ELEMS / NUM_THREADS_PER_BLOCK), 1, 1);

  mapToBuckets<<<dimGrid, dimBlock>>>(d_srcArr, d_indicesBuckets, d_bucketCounters, NUM_ELEMS, numOfBuckets);

  checkCudaErrors(cudaMemcpy(h_bucketCounters, d_bucketCounters, sizeof(int)*numOfBuckets, cudaMemcpyDeviceToHost));

  for (int i=0; i < NUM_ELEMS; i++)
  {
    cpuBucketCounters[h_srcArr[i]]++;
  }

  bool allMatch = true;
  int finalElems = 0;
  for (int i=0; i < numOfBuckets; i++)
  {
    finalElems += (h_bucketCounters[i] - i*NUM_ELEMS);
    if (cpuBucketCounters[i] != (h_bucketCounters[i] - i*NUM_ELEMS))
    {
      allMatch = false;
      break;
    }
  }

  if (!allMatch && finalElems != NUM_ELEMS)
  {
      return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  int *data_to_filter, *filtered_data, nres = 0;
  int *d_data_to_filter, *d_filtered_data, *d_nres;

  int numOfBuckets = 5;

  data_to_filter = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

  // Generate input data.
  for (int i = 0; i < NUM_ELEMS; i++) {
    data_to_filter[i] = rand() % numOfBuckets;
  }

  int devId = findCudaDevice(argc, (const char **)argv);

  checkCudaErrors(cudaMalloc(&d_data_to_filter, sizeof(int) * NUM_ELEMS));
  checkCudaErrors(cudaMalloc(&d_filtered_data, sizeof(int) * NUM_ELEMS));
  checkCudaErrors(cudaMalloc(&d_nres, sizeof(int)));

  checkCudaErrors(cudaMemcpy(d_data_to_filter, data_to_filter,
                             sizeof(int) * NUM_ELEMS, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_nres, 0, sizeof(int)));

  dim3 dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
  dim3 dimGrid((NUM_ELEMS / NUM_THREADS_PER_BLOCK) + 1, 1, 1);

  filter_arr<<<dimGrid, dimBlock>>>(d_filtered_data, d_nres, d_data_to_filter,
                                    NUM_ELEMS);

  checkCudaErrors(
      cudaMemcpy(&nres, d_nres, sizeof(int), cudaMemcpyDeviceToHost));

  filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * nres));

  checkCudaErrors(cudaMemcpy(filtered_data, d_filtered_data, sizeof(int) * nres,
                             cudaMemcpyDeviceToHost));

  int *host_filtered_data =
      reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

  // Generate host output with host filtering code.
  int host_flt_count = 0;
  for (int i = 0; i < NUM_ELEMS; i++) {
    if (data_to_filter[i] > 0) {
      host_filtered_data[host_flt_count++] = data_to_filter[i];
    }
  }

  int major = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devId));

  int mapIndicesToBucketsStatus = EXIT_SUCCESS;
  // atomicAggIncMulti require a GPU of Volta (SM7X) architecture or higher,
  // so that it can take advantage of the new MATCH capability of Volta hardware
  if (major >= 7) {
    mapIndicesToBucketsStatus = mapIndicesToBuckets(data_to_filter, d_data_to_filter, numOfBuckets);
  }

  printf("\nWarp Aggregated Atomics %s \n",
         (host_flt_count == nres) && (mapIndicesToBucketsStatus == EXIT_SUCCESS) ? "PASSED" : "FAILED");

  checkCudaErrors(cudaFree(d_data_to_filter));
  checkCudaErrors(cudaFree(d_filtered_data));
  checkCudaErrors(cudaFree(d_nres));
  free(data_to_filter);
  free(filtered_data);
  free(host_filtered_data);
}
