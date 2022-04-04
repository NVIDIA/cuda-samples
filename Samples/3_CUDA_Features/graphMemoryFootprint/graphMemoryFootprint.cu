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

// System includes
#include <assert.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#define NUM_GRAPHS 8
#define THREADS_PER_BLOCK 512

void printMemoryFootprint(int device) {
  size_t footprint;
  checkCudaErrors(cudaDeviceGetGraphMemAttribute(
      device, (cudaGraphMemAttributeType)0, &footprint));
  printf("    FOOTPRINT: %lu bytes\n", footprint);
}

void prepareAllocParams(cudaMemAllocNodeParams *allocParams, size_t bytes,
                        int device) {
  memset(allocParams, 0, sizeof(*allocParams));

  allocParams->bytesize = bytes;
  allocParams->poolProps.allocType = cudaMemAllocationTypePinned;
  allocParams->poolProps.location.id = device;
  allocParams->poolProps.location.type = cudaMemLocationTypeDevice;
}

void createVirtAddrReuseGraph(cudaGraphExec_t *graphExec, size_t bytes,
                              int device) {
  cudaGraph_t graph;
  cudaGraphNode_t allocNodeA, allocNodeB, freeNodeA, freeNodeB;
  cudaMemAllocNodeParams allocParams;
  float *d_a, *d_b;

  checkCudaErrors(cudaGraphCreate(&graph, 0));
  prepareAllocParams(&allocParams, bytes, device);

  checkCudaErrors(
      cudaGraphAddMemAllocNode(&allocNodeA, graph, NULL, 0, &allocParams));
  d_a = (float *)allocParams.dptr;
  checkCudaErrors(
      cudaGraphAddMemFreeNode(&freeNodeA, graph, &allocNodeA, 1, (void *)d_a));

  // The dependency between the allocation of d_b and the free of d_a allows d_b
  // to reuse the same VA.
  checkCudaErrors(cudaGraphAddMemAllocNode(&allocNodeB, graph, &freeNodeA, 1,
                                           &allocParams));
  d_b = (float *)allocParams.dptr;

  if (d_a == d_b) {
    printf("Check confirms that d_a and d_b share a virtual address.\n");
  } else {
    printf("Check shows that d_a and d_b DO NOT share a virtual address.\n");
  }

  checkCudaErrors(
      cudaGraphAddMemFreeNode(&freeNodeB, graph, &allocNodeB, 1, (void *)d_b));

  checkCudaErrors(cudaGraphInstantiate(graphExec, graph, NULL, NULL, 0));
  checkCudaErrors(cudaGraphDestroy(graph));
}

void virtualAddressReuseSingleGraph(size_t bytes, int device) {
  cudaStream_t stream;
  cudaGraphExec_t graphExec;

  printf("================================\n");
  printf("Running virtual address reuse example.\n");
  printf(
      "Sequential allocations & frees within a single graph enable CUDA to "
      "reuse virtual addresses.\n\n");

  createVirtAddrReuseGraph(&graphExec, bytes, device);
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  checkCudaErrors(cudaGraphLaunch(graphExec, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));
  printMemoryFootprint(device);

  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaStreamDestroy(stream));
}

// This is a kernel that does no real work but runs at least for a specified
// number of clocks
__global__ void clockBlock(clock_t clock_count) {
  unsigned int start_clock = (unsigned int)clock();

  clock_t clock_offset = 0;

  while (clock_offset < clock_count) {
    unsigned int end_clock = (unsigned int)clock();

    // The code below should work like
    // this (thanks to modular arithmetics):
    //
    // clock_offset = (clock_t) (end_clock > start_clock ?
    //                           end_clock - start_clock :
    //                           end_clock + (0xffffffffu - start_clock));
    //
    // Indeed, let m = 2^32 then
    // end - start = end + m - start (mod m).

    clock_offset = (clock_t)(end_clock - start_clock);
  }
}

// A pointer to the allocated device buffer is returned in dPtr so the caller
// can compare virtual addresses. The kernel node is added to increase the
// graph's runtime.
void createSimpleAllocFreeGraph(cudaGraphExec_t *graphExec, float **dPtr,
                                size_t bytes, int device) {
  cudaGraph_t graph;
  cudaGraphNode_t allocNodeA, freeNodeA, blockDeviceNode;
  cudaMemAllocNodeParams allocParams;
  cudaKernelNodeParams blockDeviceNodeParams = {0};
  int numElements = bytes / sizeof(float);
  float kernelTime = 5;  // time for each thread to run in microseconds

  checkCudaErrors(cudaGraphCreate(&graph, 0));
  prepareAllocParams(&allocParams, bytes, device);

  checkCudaErrors(
      cudaGraphAddMemAllocNode(&allocNodeA, graph, NULL, 0, &allocParams));
  *dPtr = (float *)allocParams.dptr;

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));
  clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);

  void *blockDeviceArgs[1] = {(void *)&time_clocks};

  size_t numBlocks = numElements / (size_t)THREADS_PER_BLOCK;
  blockDeviceNodeParams.gridDim = dim3(numBlocks, 1, 1);
  blockDeviceNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  blockDeviceNodeParams.sharedMemBytes = 0;
  blockDeviceNodeParams.extra = NULL;
  blockDeviceNodeParams.func = (void *)clockBlock;
  blockDeviceNodeParams.kernelParams = (void **)blockDeviceArgs;
  checkCudaErrors(cudaGraphAddKernelNode(&blockDeviceNode, graph, &allocNodeA,
                                         1, &blockDeviceNodeParams));

  checkCudaErrors(cudaGraphAddMemFreeNode(&freeNodeA, graph, &blockDeviceNode,
                                          1, (void *)*dPtr));

  checkCudaErrors(cudaGraphInstantiate(graphExec, graph, NULL, NULL, 0));
  checkCudaErrors(cudaGraphDestroy(graph));
}

void physicalMemoryReuseSingleStream(size_t bytes, int device) {
  cudaStream_t stream;
  cudaGraphExec_t graphExecs[NUM_GRAPHS];
  float *dPtrs[NUM_GRAPHS];
  bool virtualAddrDiffer = true;

  printf("================================\n");
  printf("Running physical memory reuse example.\n");
  printf(
      "CUDA reuses the same physical memory for allocations from separate "
      "graphs when the allocation lifetimes don't overlap.\n\n");

  for (int i = 0; i < NUM_GRAPHS; i++) {
    createSimpleAllocFreeGraph(&graphExecs[i], &dPtrs[i], bytes, device);
  }

  printf("Creating the graph execs does not reserve any physical memory.\n");
  printMemoryFootprint(device);

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  checkCudaErrors(cudaGraphLaunch(graphExecs[0], stream));
  printf("\nThe first graph launched reserves the memory it needs.\n");
  printMemoryFootprint(device);

  checkCudaErrors(cudaGraphLaunch(graphExecs[0], stream));
  printf(
      "A subsequent launch of the same graph in the same stream reuses the "
      "same physical memory. ");
  printf("Thus the memory footprint does not grow here.\n");
  printMemoryFootprint(device);

  printf(
      "\nSubsequent launches of other graphs in the same stream also reuse the "
      "physical memory. ");
  printf("Thus the memory footprint does not grow here.\n");
  for (int i = 1; i < NUM_GRAPHS; i++) {
    checkCudaErrors(cudaGraphLaunch(graphExecs[i], stream));
    printf("%02d: ", i);
    printMemoryFootprint(device);
  }

  checkCudaErrors(cudaStreamSynchronize(stream));

  for (int i = 0; i < NUM_GRAPHS; i++) {
    for (int j = i + 1; j < NUM_GRAPHS; j++) {
      if (dPtrs[i] == dPtrs[j]) {
        virtualAddrDiffer = false;
        printf("Error: Graph exec %d and %d have the same virtual address!\n",
               i - 1, i);
      }
    }
    checkCudaErrors(cudaGraphExecDestroy(graphExecs[i]));
  }
  if (virtualAddrDiffer) {
    printf("\nCheck confirms all graphs use a different virtual address.\n");
  } else {
    printf(
        "\nAll graphs do NOT use different virtual addresses. Exiting test.\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaStreamDestroy(stream));
}

void simultaneousStreams(size_t bytes, int device) {
  cudaStream_t streams[NUM_GRAPHS];
  cudaGraphExec_t graphExecs[NUM_GRAPHS];
  float *dPtrs[NUM_GRAPHS];

  printf("================================\n");
  printf("Running simultaneous streams example.\n");
  printf("Graphs that can run concurrently need separate physical memory. ");
  printf(
      "In this example, each graph launched in a separate stream increases the "
      "total memory footprint.\n\n");

  printf(
      "When launching a new graph, CUDA may reuse physical memory from a graph "
      "whose execution has already ");
  printf(
      "finished -- even if the new graph is being launched in a different "
      "stream from the completed graph. ");
  printf(
      "Therefore, a kernel node is added to the graphs to increase "
      "runtime.\n\n");

  for (int i = 0; i < NUM_GRAPHS; i++) {
    createSimpleAllocFreeGraph(&graphExecs[i], &dPtrs[i], bytes, device);
    checkCudaErrors(
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
  }

  printf("Initial footprint:\n");
  printMemoryFootprint(device);

  printf(
      "\nEach graph launch in a seperate stream grows the memory footprint:\n");
  for (int i = 1; i < NUM_GRAPHS; i++) {
    checkCudaErrors(cudaGraphLaunch(graphExecs[i], streams[i]));
    printf("%02d: ", i);
    printMemoryFootprint(device);
  }

  for (int i = 0; i < NUM_GRAPHS; i++) {
    checkCudaErrors(cudaStreamSynchronize(streams[i]));
    checkCudaErrors(cudaGraphExecDestroy(graphExecs[i]));
    checkCudaErrors(cudaStreamDestroy(streams[i]));
  }
}

void createSimpleAllocNoFreeGraph(cudaGraphExec_t *graphExec, float **dPtr,
                                  size_t bytes, int device) {
  cudaGraph_t graph;
  cudaGraphNode_t allocNodeA;
  cudaMemAllocNodeParams allocParams;

  checkCudaErrors(cudaGraphCreate(&graph, 0));
  prepareAllocParams(&allocParams, bytes, device);

  checkCudaErrors(
      cudaGraphAddMemAllocNode(&allocNodeA, graph, NULL, 0, &allocParams));
  *dPtr = (float *)allocParams.dptr;

  checkCudaErrors(cudaGraphInstantiate(graphExec, graph, NULL, NULL, 0));
  checkCudaErrors(cudaGraphDestroy(graph));
}

void unfreedAllocations(size_t bytes, int device) {
  cudaStream_t stream;
  cudaGraphExec_t graphExecs[NUM_GRAPHS];
  float *dPtrs[NUM_GRAPHS];

  printf("================================\n");
  printf("Running unfreed streams example.\n");
  printf(
      "CUDA cannot reuse phyiscal memory from graphs which do not free their "
      "allocations.\n\n");

  for (int i = 0; i < NUM_GRAPHS; i++) {
    createSimpleAllocNoFreeGraph(&graphExecs[i], &dPtrs[i], bytes, device);
  }

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  printf(
      "Despite being launched in the same stream, each graph launch grows the "
      "memory footprint. ");
  printf(
      "Since the allocation is not freed, CUDA keeps the memory valid for "
      "use.\n");
  for (int i = 0; i < NUM_GRAPHS; i++) {
    checkCudaErrors(cudaGraphLaunch(graphExecs[i], stream));
    printf("%02d: ", i);
    printMemoryFootprint(device);
  }

  checkCudaErrors(cudaStreamSynchronize(stream));

  checkCudaErrors(cudaDeviceGraphMemTrim(device));
  printf(
      "\nTrimming does not impact the memory footprint since the un-freed "
      "allocations are still holding onto the memory.\n");
  printMemoryFootprint(device);

  for (int i = 0; i < NUM_GRAPHS; i++) {
    checkCudaErrors(cudaFree(dPtrs[i]));
  }
  printf("\nFreeing the allocations does not shrink the footprint.\n");
  printMemoryFootprint(device);

  checkCudaErrors(cudaDeviceGraphMemTrim(device));
  printf(
      "\nSince the allocations are now freed, trimming does reduce the "
      "footprint even when the graph execs are not yet destroyed.\n");
  printMemoryFootprint(device);

  for (int i = 0; i < NUM_GRAPHS; i++) {
    checkCudaErrors(cudaGraphExecDestroy(graphExecs[i]));
  }
  checkCudaErrors(cudaStreamDestroy(stream));
}

void cleanupMemory(int device) {
  checkCudaErrors(cudaDeviceGraphMemTrim(device));
  printf("\nCleaning up example by trimming device memory.\n");
  printMemoryFootprint(device);
  printf("\n");
}

int main(int argc, char **argv) {
  size_t bytes = 64 * 1024 * 1024;
  int device = findCudaDevice(argc, (const char **)argv);

  int driverVersion = 0;
  int deviceSupportsMemoryPools = 0;

  cudaDriverGetVersion(&driverVersion);
  printf("Driver version is: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);

  if (driverVersion < 11040) {
    printf("Waiving execution as driver does not support Graph Memory Nodes\n");
    exit(EXIT_WAIVED);
  }

  cudaDeviceGetAttribute(&deviceSupportsMemoryPools,
                         cudaDevAttrMemoryPoolsSupported, device);
  if (!deviceSupportsMemoryPools) {
    printf("Waiving execution as device does not support Memory Pools\n");
    exit(EXIT_WAIVED);
  } else {
    printf("Running sample.\n");
  }

  virtualAddressReuseSingleGraph(bytes, device);
  cleanupMemory(device);

  physicalMemoryReuseSingleStream(bytes, device);
  cleanupMemory(device);

  simultaneousStreams(bytes, device);
  cleanupMemory(device);

  unfreedAllocations(bytes, device);
  cleanupMemory(device);

  printf("================================\n");
  printf("Sample complete.\n");
}
