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

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <vector>

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 3

typedef struct callBackData {
  const char *fn_name;
  double *data;
} callBackData_t;

__global__ void reduce(float *inputVec, double *outputVec, size_t inputSize,
                       size_t outputSize) {
  __shared__ double tmp[THREADS_PER_BLOCK];

  cg::thread_block cta = cg::this_thread_block();
  size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

  double temp_sum = 0.0;
  for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
    temp_sum += (double)inputVec[i];
  }
  tmp[cta.thread_rank()] = temp_sum;

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  double beta = temp_sum;
  double temp;

  for (int i = tile32.size() / 2; i > 0; i >>= 1) {
    if (tile32.thread_rank() < i) {
      temp = tmp[cta.thread_rank() + i];
      beta += temp;
      tmp[cta.thread_rank()] = beta;
    }
    cg::sync(tile32);
  }
  cg::sync(cta);

  if (cta.thread_rank() == 0 && blockIdx.x < outputSize) {
    beta = 0.0;
    for (int i = 0; i < cta.size(); i += tile32.size()) {
      beta += tmp[i];
    }
    outputVec[blockIdx.x] = beta;
  }
}

__global__ void reduceFinal(double *inputVec, double *result,
                            size_t inputSize) {
  __shared__ double tmp[THREADS_PER_BLOCK];

  cg::thread_block cta = cg::this_thread_block();
  size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

  double temp_sum = 0.0;
  for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
    temp_sum += (double)inputVec[i];
  }
  tmp[cta.thread_rank()] = temp_sum;

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  // do reduction in shared mem
  if ((blockDim.x >= 512) && (cta.thread_rank() < 256)) {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 256];
  }

  cg::sync(cta);

  if ((blockDim.x >= 256) && (cta.thread_rank() < 128)) {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 128];
  }

  cg::sync(cta);

  if ((blockDim.x >= 128) && (cta.thread_rank() < 64)) {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 64];
  }

  cg::sync(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockDim.x >= 64) temp_sum += tmp[cta.thread_rank() + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      temp_sum += tile32.shfl_down(temp_sum, offset);
    }
  }
  // write result for this block to global mem
  if (cta.thread_rank() == 0) result[0] = temp_sum;
}

void init_input(float *a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

void CUDART_CB myHostNodeCallback(void *data) {
  // Check status of GPU after stream operations are done
  callBackData_t *tmp = (callBackData_t *)(data);
  // checkCudaErrors(tmp->status);

  double *result = (double *)(tmp->data);
  char *function = (char *)(tmp->fn_name);
  printf("[%s] Host callback final reduced sum = %lf\n", function, *result);
  *result = 0.0;  // reset the result
}

void cudaGraphsManual(float *inputVec_h, float *inputVec_d, double *outputVec_d,
                      double *result_d, size_t inputSize, size_t numOfBlocks) {
  cudaStream_t streamForGraph;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> nodeDependencies;
  cudaGraphNode_t memcpyNode, kernelNode, memsetNode;
  double result_h = 0.0;

  checkCudaErrors(cudaStreamCreate(&streamForGraph));

  cudaKernelNodeParams kernelNodeParams = {0};
  cudaMemcpy3DParms memcpyParams = {0};
  cudaMemsetParams memsetParams = {0};

  memcpyParams.srcArray = NULL;
  memcpyParams.srcPos = make_cudaPos(0, 0, 0);
  memcpyParams.srcPtr =
      make_cudaPitchedPtr(inputVec_h, sizeof(float) * inputSize, inputSize, 1);
  memcpyParams.dstArray = NULL;
  memcpyParams.dstPos = make_cudaPos(0, 0, 0);
  memcpyParams.dstPtr =
      make_cudaPitchedPtr(inputVec_d, sizeof(float) * inputSize, inputSize, 1);
  memcpyParams.extent = make_cudaExtent(sizeof(float) * inputSize, 1, 1);
  memcpyParams.kind = cudaMemcpyHostToDevice;

  memsetParams.dst = (void *)outputVec_d;
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(float);  // elementSize can be max 4 bytes
  memsetParams.width = numOfBlocks * 2;
  memsetParams.height = 1;

  checkCudaErrors(cudaGraphCreate(&graph, 0));
  checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams));
  checkCudaErrors(
      cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

  nodeDependencies.push_back(memsetNode);
  nodeDependencies.push_back(memcpyNode);

  void *kernelArgs[4] = {(void *)&inputVec_d, (void *)&outputVec_d, &inputSize,
                         &numOfBlocks};

  kernelNodeParams.func = (void *)reduce;
  kernelNodeParams.gridDim = dim3(numOfBlocks, 1, 1);
  kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = (void **)kernelArgs;
  kernelNodeParams.extra = NULL;

  checkCudaErrors(
      cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams));

  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = result_d;
  memsetParams.value = 0;
  memsetParams.elementSize = sizeof(float);
  memsetParams.width = 2;
  memsetParams.height = 1;
  checkCudaErrors(
      cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

  nodeDependencies.push_back(memsetNode);

  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func = (void *)reduceFinal;
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  void *kernelArgs2[3] = {(void *)&outputVec_d, (void *)&result_d,
                          &numOfBlocks};
  kernelNodeParams.kernelParams = kernelArgs2;
  kernelNodeParams.extra = NULL;

  checkCudaErrors(
      cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);

  memset(&memcpyParams, 0, sizeof(memcpyParams));

  memcpyParams.srcArray = NULL;
  memcpyParams.srcPos = make_cudaPos(0, 0, 0);
  memcpyParams.srcPtr = make_cudaPitchedPtr(result_d, sizeof(double), 1, 1);
  memcpyParams.dstArray = NULL;
  memcpyParams.dstPos = make_cudaPos(0, 0, 0);
  memcpyParams.dstPtr = make_cudaPitchedPtr(&result_h, sizeof(double), 1, 1);
  memcpyParams.extent = make_cudaExtent(sizeof(double), 1, 1);
  memcpyParams.kind = cudaMemcpyDeviceToHost;
  checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &memcpyParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(memcpyNode);

  cudaGraphNode_t hostNode;
  cudaHostNodeParams hostParams = {0};
  hostParams.fn = myHostNodeCallback;
  callBackData_t hostFnData;
  hostFnData.data = &result_h;
  hostFnData.fn_name = "cudaGraphsManual";
  hostParams.userData = &hostFnData;

  checkCudaErrors(cudaGraphAddHostNode(&hostNode, graph,
                                       nodeDependencies.data(),
                                       nodeDependencies.size(), &hostParams));

  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  checkCudaErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
  printf("\nNum of nodes in the graph created manually = %zu\n", numNodes);

  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  cudaGraph_t clonedGraph;
  cudaGraphExec_t clonedGraphExec;
  checkCudaErrors(cudaGraphClone(&clonedGraph, graph));
  checkCudaErrors(
      cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));

  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
  }

  checkCudaErrors(cudaStreamSynchronize(streamForGraph));

  printf("Cloned Graph Output.. \n");
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    checkCudaErrors(cudaGraphLaunch(clonedGraphExec, streamForGraph));
  }
  checkCudaErrors(cudaStreamSynchronize(streamForGraph));

  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaGraphExecDestroy(clonedGraphExec));
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaGraphDestroy(clonedGraph));
  checkCudaErrors(cudaStreamDestroy(streamForGraph));
}

void cudaGraphsUsingStreamCapture(float *inputVec_h, float *inputVec_d,
                                  double *outputVec_d, double *result_d,
                                  size_t inputSize, size_t numOfBlocks) {
  cudaStream_t stream1, stream2, stream3, streamForGraph;
  cudaEvent_t forkStreamEvent, memsetEvent1, memsetEvent2;
  cudaGraph_t graph;
  double result_h = 0.0;

  checkCudaErrors(cudaStreamCreate(&stream1));
  checkCudaErrors(cudaStreamCreate(&stream2));
  checkCudaErrors(cudaStreamCreate(&stream3));
  checkCudaErrors(cudaStreamCreate(&streamForGraph));

  checkCudaErrors(cudaEventCreate(&forkStreamEvent));
  checkCudaErrors(cudaEventCreate(&memsetEvent1));
  checkCudaErrors(cudaEventCreate(&memsetEvent2));

  checkCudaErrors(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

  checkCudaErrors(cudaEventRecord(forkStreamEvent, stream1));
  checkCudaErrors(cudaStreamWaitEvent(stream2, forkStreamEvent, 0));
  checkCudaErrors(cudaStreamWaitEvent(stream3, forkStreamEvent, 0));

  checkCudaErrors(cudaMemcpyAsync(inputVec_d, inputVec_h,
                                  sizeof(float) * inputSize, cudaMemcpyDefault,
                                  stream1));

  checkCudaErrors(
      cudaMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks, stream2));

  checkCudaErrors(cudaEventRecord(memsetEvent1, stream2));

  checkCudaErrors(cudaMemsetAsync(result_d, 0, sizeof(double), stream3));
  checkCudaErrors(cudaEventRecord(memsetEvent2, stream3));

  checkCudaErrors(cudaStreamWaitEvent(stream1, memsetEvent1, 0));

  reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream1>>>(
      inputVec_d, outputVec_d, inputSize, numOfBlocks);

  checkCudaErrors(cudaStreamWaitEvent(stream1, memsetEvent2, 0));

  reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_d, result_d,
                                                    numOfBlocks);
  checkCudaErrors(cudaMemcpyAsync(&result_h, result_d, sizeof(double),
                                  cudaMemcpyDefault, stream1));

  callBackData_t hostFnData = {0};
  hostFnData.data = &result_h;
  hostFnData.fn_name = "cudaGraphsUsingStreamCapture";
  cudaHostFn_t fn = myHostNodeCallback;
  checkCudaErrors(cudaLaunchHostFunc(stream1, fn, &hostFnData));
  checkCudaErrors(cudaStreamEndCapture(stream1, &graph));

  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  checkCudaErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
  printf("\nNum of nodes in the graph created using stream capture API = %zu\n",
         numNodes);

  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  cudaGraph_t clonedGraph;
  cudaGraphExec_t clonedGraphExec;
  checkCudaErrors(cudaGraphClone(&clonedGraph, graph));
  checkCudaErrors(
      cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));

  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
  }

  checkCudaErrors(cudaStreamSynchronize(streamForGraph));

  printf("Cloned Graph Output.. \n");
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    checkCudaErrors(cudaGraphLaunch(clonedGraphExec, streamForGraph));
  }

  checkCudaErrors(cudaStreamSynchronize(streamForGraph));

  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaGraphExecDestroy(clonedGraphExec));
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaGraphDestroy(clonedGraph));
  checkCudaErrors(cudaStreamDestroy(stream1));
  checkCudaErrors(cudaStreamDestroy(stream2));
  checkCudaErrors(cudaStreamDestroy(streamForGraph));
}

int main(int argc, char **argv) {
  size_t size = 1 << 24;  // number of elements to reduce
  size_t maxBlocks = 512;

  // This will pick the best possible CUDA capable device
  int devID = findCudaDevice(argc, (const char **)argv);

  printf("%zu elements\n", size);
  printf("threads per block  = %d\n", THREADS_PER_BLOCK);
  printf("Graph Launch iterations = %d\n", GRAPH_LAUNCH_ITERATIONS);

  float *inputVec_d = NULL, *inputVec_h = NULL;
  double *outputVec_d = NULL, *result_d;

  inputVec_h = (float *)malloc(sizeof(float) * size);
  checkCudaErrors(cudaMalloc(&inputVec_d, sizeof(float) * size));
  checkCudaErrors(cudaMalloc(&outputVec_d, sizeof(double) * maxBlocks));
  checkCudaErrors(cudaMalloc(&result_d, sizeof(double)));

  init_input(inputVec_h, size);

  cudaGraphsManual(inputVec_h, inputVec_d, outputVec_d, result_d, size,
                   maxBlocks);
  cudaGraphsUsingStreamCapture(inputVec_h, inputVec_d, outputVec_d, result_d,
                               size, maxBlocks);

  checkCudaErrors(cudaFree(inputVec_d));
  checkCudaErrors(cudaFree(outputVec_d));
  checkCudaErrors(cudaFree(result_d));
  return EXIT_SUCCESS;
}
