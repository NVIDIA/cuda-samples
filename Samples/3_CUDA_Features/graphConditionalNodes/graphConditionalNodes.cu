/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
 * This file demonstrates the usage of conditional graph nodes with
 * a series of *simple* example graphs.
 * 
 * For more information on conditional nodes, see the programming guide:
 * 
 *   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#conditional-graph-nodes
 *
 */

// System includes
#include <assert.h>
#include <stdio.h>

#include <climits>
#include <vector>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

/*
 * Create a graph containing two nodes.
 * The first node, A, is a kernel and the second node, B, is a conditional IF node.
 * The kernel sets the condition variable to true if a device memory location
 * contains an odd number. Otherwise the condition variable is set to false.
 * There is a single kernel, C, within the conditional body which prints a message.
 *
 * A -> B [ C ]
 *
 */

__global__ void ifGraphKernelA(char *dPtr, cudaGraphConditionalHandle handle)
{
	// In this example, condition is set if *dPtr is odd
    unsigned int value = *dPtr & 0x01;
    cudaGraphSetConditional(handle, value);
    printf("GPU: Handle set to %d\n", value);
}

// This kernel will only be executed if the condition is true
__global__ void ifGraphKernelC(void)
{
    printf("GPU: Hello from the GPU!\n");
}

// Setup and launch the graph
void simpleIfGraph(void)
{
    cudaGraph_t     graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;

    void *kernelArgs[2];
    char *dPtr;      // Pointer to device memory location

    // Allocate a byte of device memory to use as input
    checkCudaErrors(cudaMalloc((void**)&dPtr, 1));

    cudaGraphCreate(&graph, 0);

    // Create conditional handle.
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // Use a kernel upstream of the conditional to set the handle value
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func         = (void *)ifGraphKernelA;
    params.kernel.gridDim.x    = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x   = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &dPtr;
    kernelArgs[1] = &handle;
    checkCudaErrors(cudaGraphAddNode(&node, graph, NULL, 0, &params));

    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeIf;
    cParams.conditional.size   = 1;
    checkCudaErrors(cudaGraphAddNode(&node, graph, &node, 1, &cParams));

    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    // Populate the body of the conditional node
    cudaGraphNode_t bodyNode;
    params.kernel.func         = (void *)ifGraphKernelC;
    params.kernel.kernelParams = nullptr;
    checkCudaErrors(cudaGraphAddNode(&bodyNode, bodyGraph, NULL, 0, &params));

    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Initialize device memory and launch the graph
    checkCudaErrors(cudaMemset(dPtr, 0, 1)); // Set dPtr to 0
    printf("Host: Launching graph with conditional value set to false\n");
    checkCudaErrors(cudaGraphLaunch(graphExec, 0));
    checkCudaErrors(cudaDeviceSynchronize());

    // Initialize device memory and launch the graph
    checkCudaErrors(cudaMemset(dPtr, 1, 1)); // Set dPtr to 1
    printf("Host: Launching graph with conditional value set to true\n");
    checkCudaErrors(cudaGraphLaunch(graphExec, 0));
    checkCudaErrors(cudaDeviceSynchronize());

    // Cleanup
    checkCudaErrors(cudaGraphExecDestroy(graphExec));
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaFree(dPtr));
}

int main(int argc, char **argv) {
    int device = findCudaDevice(argc, (const char **)argv);

    int driverVersion = 0;

    cudaDriverGetVersion(&driverVersion);
    printf("Driver version is: %d.%d\n", driverVersion / 1000,
            (driverVersion % 100) / 10);

    if (driverVersion < 12030) {
        printf("Waiving execution as driver does not support Graph Conditional Nodes\n");
        exit(EXIT_WAIVED);
    }

    simpleIfGraph();

    return 0;
}
