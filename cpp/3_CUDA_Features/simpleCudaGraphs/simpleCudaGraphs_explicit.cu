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
 * Demonstrates explicit CUDA Graph construction using the unified cudaGraphAddNode API.
 *
 * Nodes and edges are added one at a time to build this graph:
 *
 *   memcpy (H2D) -> reduce -> reduceFinal -> memcpy (D2H) -> host callback
 *
 * The workload is a two-pass reduction: an input float vector is reduced to
 * per-block partial sums (reduce), then to a single double result (reduceFinal).
 * The graph is instantiated once and launched GRAPH_LAUNCH_ITERATIONS times,
 * then cloned and launched again.
 */

#include "simpleCudaGraphs.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define GRAPH_LAUNCH_ITERATIONS 3

void cudaGraphsManual(float  *inputVec_h,
                      float  *inputVec_d,
                      double *outputVec_d,
                      double *result_d,
                      size_t  inputSize,
                      size_t  numOfBlocks)
{
    cudaStream_t                 streamForGraph;
    cudaGraph_t                  graph;
    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t              memcpyNode, kernelNode;
    double                       result_h = 0.0;

    cudaStreamCreate(&streamForGraph);

    // cudaGraphAddNode is the unified node-creation API: one call for every node type,
    // driven by a cudaGraphNodeParams struct (a .type tag + a union of per-type params).
    // It replaces the previously used cudaGraphAddMemcpyNode/cudaGraphAddKernelNode calls. The struct
    // has reserved fields that must be zero, so it is re-zeroed with "= {}" before each node below.
    cudaGraphNodeParams nodeParams   = {};
    cudaMemcpy3DParms   memcpyParams = {0};

    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr   = make_cudaPitchedPtr(inputVec_h, sizeof(float) * inputSize, inputSize, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr   = make_cudaPitchedPtr(inputVec_d, sizeof(float) * inputSize, inputSize, 1);
    memcpyParams.extent   = make_cudaExtent(sizeof(float) * inputSize, 1, 1);
    memcpyParams.kind     = cudaMemcpyHostToDevice;

    // Create an empty graph; nodes and edges will be added below.
    cudaGraphCreate(&graph, 0);

    // Node 1: H2D memcpy — no dependencies (NULL, 0), so it can start immediately.
    // For a memcpy node the cudaMemcpy3DParms goes into nodeParams.memcpy.copyParams
    // (the .memcpy union member is a wrapper struct, so the copy descriptor nests one level deeper).
    nodeParams                   = {};
    nodeParams.type              = cudaGraphNodeTypeMemcpy;
    nodeParams.memcpy.copyParams = memcpyParams;
    cudaGraphAddNode(&memcpyNode, graph, NULL, /*dependencyData=*/NULL, 0, &nodeParams);

    // Make the next node wait for this memcpy to finish before starting.
    nodeDependencies.push_back(memcpyNode);

    void *kernelArgs[4] = {(void *)&inputVec_d, (void *)&outputVec_d, &inputSize, &numOfBlocks};

    // Node 2: first reduction kernel — depends on the H2D memcpy completing.
    // Kernel params are set directly on the .kernel union member.
    nodeParams                       = {};
    nodeParams.type                  = cudaGraphNodeTypeKernel;
    nodeParams.kernel.func           = (void *)reduce;
    nodeParams.kernel.gridDim        = dim3(numOfBlocks, 1, 1);
    nodeParams.kernel.blockDim       = dim3(THREADS_PER_BLOCK, 1, 1);
    nodeParams.kernel.sharedMemBytes = 0;
    nodeParams.kernel.kernelParams   = (void **)kernelArgs;
    nodeParams.kernel.extra          = NULL;
    cudaGraphAddNode(
        &kernelNode, graph, nodeDependencies.data(), /*dependencyData=*/NULL, nodeDependencies.size(), &nodeParams);

    // Move dependency forward: the next node waits for this kernel.
    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);

    void *kernelArgs2[3] = {(void *)&outputVec_d, (void *)&result_d, &numOfBlocks};

    // Node 3: final reduction kernel — depends on Node 2.
    nodeParams                       = {};
    nodeParams.type                  = cudaGraphNodeTypeKernel;
    nodeParams.kernel.func           = (void *)reduceFinal;
    nodeParams.kernel.gridDim        = dim3(1, 1, 1);
    nodeParams.kernel.blockDim       = dim3(THREADS_PER_BLOCK, 1, 1);
    nodeParams.kernel.sharedMemBytes = 0;
    nodeParams.kernel.kernelParams   = kernelArgs2;
    nodeParams.kernel.extra          = NULL;
    cudaGraphAddNode(
        &kernelNode, graph, nodeDependencies.data(), /*dependencyData=*/NULL, nodeDependencies.size(), &nodeParams);
    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);

    memset(&memcpyParams, 0, sizeof(memcpyParams));

    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr   = make_cudaPitchedPtr(result_d, sizeof(double), 1, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr   = make_cudaPitchedPtr(&result_h, sizeof(double), 1, 1);
    memcpyParams.extent   = make_cudaExtent(sizeof(double), 1, 1);
    memcpyParams.kind     = cudaMemcpyDeviceToHost;

    // Node 4: D2H memcpy — copies the scalar result back to the host.
    // Again the copy descriptor nests in nodeParams.memcpy.copyParams.
    nodeParams                   = {};
    nodeParams.type              = cudaGraphNodeTypeMemcpy;
    nodeParams.memcpy.copyParams = memcpyParams;
    cudaGraphAddNode(&memcpyNode, graph, nodeDependencies.data(), /*dependencyData=*/NULL, nodeDependencies.size(), &nodeParams);
    nodeDependencies.clear();
    nodeDependencies.push_back(memcpyNode);

    cudaGraphNode_t hostNode;
    callBackData_t  hostFnData;
    hostFnData.data    = &result_h;
    hostFnData.fn_name = "cudaGraphsManual";

    // Node 5: host callback — runs on the CPU after the D2H copy completes.
    // The .host member is cudaHostNodeParamsV2 (fn + userData, plus a syncMode field
    // left at 0 by the zero-init above).
    nodeParams               = {};
    nodeParams.type          = cudaGraphNodeTypeHost;
    nodeParams.host.fn       = myHostNodeCallback;
    nodeParams.host.userData = &hostFnData;
    cudaGraphAddNode(&hostNode, graph, nodeDependencies.data(), /*dependencyData=*/NULL, nodeDependencies.size(), &nodeParams);

    size_t numNodes = 0;
    cudaGraphGetNodes(graph, NULL, &numNodes);
    printf("Graph node count: %zu\n", numNodes);

    // Instantiate: compile the graph into an executable form (one-time cost).
    // This is where CUDA optimizes the schedule; repeated launches reuse this.
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Demonstrates cudaGraphClone — in practice, cloning is useful when multiple CPU
    // threads need to launch the same graph concurrently, each with its own independent graphExec.
    cudaGraph_t     clonedGraph;
    cudaGraphExec_t clonedGraphExec;
    cudaGraphClone(&clonedGraph, graph);
    cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0);

    // Refill the host input before each launch so the graph's H2D copy processes
    // new data every time — one instantiated graph reused for different data.
    // The per-iteration sync ensures that copy finishes before we overwrite the buffer.
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        init_input(inputVec_h, inputSize);
        cudaGraphLaunch(graphExec, streamForGraph);
        cudaStreamSynchronize(streamForGraph);
    }

    printf("\nCloned graph:\n");
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        init_input(inputVec_h, inputSize);
        cudaGraphLaunch(clonedGraphExec, streamForGraph);
        cudaStreamSynchronize(streamForGraph);
    }

    cudaGraphExecDestroy(graphExec);
    cudaGraphExecDestroy(clonedGraphExec);
    cudaGraphDestroy(graph);
    cudaGraphDestroy(clonedGraph);
    cudaStreamDestroy(streamForGraph);
}

int main()
{
    size_t size      = 1 << 24; // number of elements to reduce, 16M elements
    size_t maxBlocks = 512;

    int devID = 0;
    cudaSetDevice(devID);

    int major, minor, smCount;
    cudaDeviceGetAttribute(&major,   cudaDevAttrComputeCapabilityMajor, devID);
    cudaDeviceGetAttribute(&minor,   cudaDevAttrComputeCapabilityMinor, devID);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount,    devID);
    printf("GPU Device %d: compute capability %d.%d, %d SMs\n\n", devID, major, minor, smCount);

    printf("Reducing %zu elements\n", size);
    printf("Threads per block   : %d\n", THREADS_PER_BLOCK);
    printf("Graph launch iterations: %d\n\n", GRAPH_LAUNCH_ITERATIONS);

    float  *inputVec_h = NULL, *inputVec_d = NULL;
    double *outputVec_d = NULL, *result_d = NULL;

    cudaMallocHost(&inputVec_h, sizeof(float) * size);
    cudaMalloc(&inputVec_d, sizeof(float) * size);
    cudaMalloc(&outputVec_d, sizeof(double) * maxBlocks);
    cudaMalloc(&result_d, sizeof(double));

    printf("=== Explicit Graph Construction ===\n");
    cudaGraphsManual(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);

    cudaFree(inputVec_d);
    cudaFree(outputVec_d);
    cudaFree(result_d);
    cudaFreeHost(inputVec_h);
    return EXIT_SUCCESS;
}
