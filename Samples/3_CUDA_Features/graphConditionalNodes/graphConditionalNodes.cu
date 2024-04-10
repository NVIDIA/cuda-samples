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
#include <cassert>
#include <cstdio>

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

    // Allocate a byte of device memory to use as input
    char *dPtr;
    checkCudaErrors(cudaMalloc((void**)&dPtr, 1));

    printf("simpleIfGraph: Building graph...\n");
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

    printf("simpleIfGraph: Complete\n\n");
}

/*
 * Create a graph containing a single conditional while node.
 * The default value of the conditional variable is set to true, so this
 * effectively becomes a do-while loop as the conditional body will always
 * execute at least once. The body of the conditional contains 3 kernel nodes:
 * A [ B -> C -> D ]
 * Nodes B and C are just dummy nodes for demonstrative purposes. Node D
 * will decrement a device memory location and set the condition value to false
 * when the value reaches zero, terminating the loop.
 * In this example, stream capture is used to populate the conditional body.
 */

// This kernel will only be executed if the condition is true
__global__ void doWhileEmptyKernel(void)
{
    printf("GPU: doWhileEmptyKernel()\n");
    return;
}

__global__ void doWhileLoopKernel(char *dPtr, cudaGraphConditionalHandle handle)
{
    if (--(*dPtr) == 0) {
        cudaGraphSetConditional(handle, 0);
    }
    printf("GPU: counter = %d\n", *dPtr);
}

void simpleDoWhileGraph(void)
{
    cudaGraph_t     graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;

    // Allocate a byte of device memory to use as input
    char *dPtr;
    checkCudaErrors(cudaMalloc((void**)&dPtr, 1));

    printf("simpleDoWhileGraph: Building graph...\n");
    checkCudaErrors(cudaGraphCreate(&graph, 0));

    cudaGraphConditionalHandle handle;
    checkCudaErrors(cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault));

    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeWhile;
    cParams.conditional.size   = 1;
    checkCudaErrors(cudaGraphAddNode(&node, graph, NULL, 0, &cParams));

    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    cudaStream_t captureStream;
    checkCudaErrors(cudaStreamCreate(&captureStream));
    
    checkCudaErrors(cudaStreamBeginCaptureToGraph(captureStream, bodyGraph, nullptr, nullptr, 0, cudaStreamCaptureModeRelaxed));
    doWhileEmptyKernel<<<1, 1, 0, captureStream>>>();
    doWhileEmptyKernel<<<1, 1, 0, captureStream>>>();
    doWhileLoopKernel<<<1, 1, 0, captureStream>>>(dPtr, handle);
    checkCudaErrors(cudaStreamEndCapture(captureStream, nullptr));
    checkCudaErrors(cudaStreamDestroy(captureStream));

    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Initialize device memory and launch the graph
    checkCudaErrors(cudaMemset(dPtr, 10, 1)); // Set dPtr to 10
    printf("Host: Launching graph with loop counter set to 10\n");
    checkCudaErrors(cudaGraphLaunch(graphExec, 0));
    checkCudaErrors(cudaDeviceSynchronize());

    // Cleanup
    checkCudaErrors(cudaGraphExecDestroy(graphExec));
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaFree(dPtr));

    printf("simpleDoWhileGraph: Complete\n\n");
}


/*
 * Create a graph containing a conditional while loop using stream capture.
 * This demonstrates how to insert a conditional node into a stream which is
 * being captured. The graph consists of a kernel node followed by a conditional
 * while node which contains a single kernel node:
 *
 * A -> B [ C ]
 *
 * The same kernel will be used for both nodes A and C. This kernel will test
 * a device memory location and set the condition when the location is non-zero.
 * We must run the kernel before the loop as well as inside the loop in order
 * to behave like a while loop. We need to evaluate the device memory location
 * before the conditional node is evaluated in order to set the condition variable
 * properly. Because we're using a kernel upstream of the conditional node,
 * there is no need to use the handle default value to initialize the conditional
 * value.
 */

__global__ void capturedWhileKernel(char *dPtr, cudaGraphConditionalHandle handle)
{
    printf("GPU: counter = %d\n", *dPtr);
    if (*dPtr) {
        (*dPtr)--;
    }
    cudaGraphSetConditional(handle, *dPtr);
}

__global__ void capturedWhileEmptyKernel(void)
{
    printf("GPU: capturedWhileEmptyKernel()\n");
    return;
}

void capturedWhileGraph(void)
{
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cudaStreamCaptureStatus status;
    const cudaGraphNode_t *dependencies;
    size_t numDependencies;

    // Allocate a byte of device memory to use as input
    char *dPtr;
    checkCudaErrors(cudaMalloc((void**)&dPtr, 1));

    printf("capturedWhileGraph: Building graph...\n");
    cudaStream_t captureStream;
    checkCudaErrors(cudaStreamCreate(&captureStream));

    checkCudaErrors(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeRelaxed));

    // Obtain the handle of the graph
    checkCudaErrors(cudaStreamGetCaptureInfo(captureStream, &status, NULL, &graph, &dependencies, &numDependencies));

    // Create the conditional handle
    cudaGraphConditionalHandle handle;
    checkCudaErrors(cudaGraphConditionalHandleCreate(&handle, graph));

    // Insert kernel node A
    capturedWhileKernel<<<1, 1, 0, captureStream>>>(dPtr, handle);

    // Obtain the handle for node A
    checkCudaErrors(cudaStreamGetCaptureInfo(captureStream, &status, NULL, &graph, &dependencies, &numDependencies));

    // Insert conditional node B
    cudaGraphNode_t node;
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeWhile;
    cParams.conditional.size   = 1;
    checkCudaErrors(cudaGraphAddNode(&node, graph, dependencies, numDependencies, &cParams));

    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    // Update stream capture dependencies to account for the node we manually added
    checkCudaErrors(cudaStreamUpdateCaptureDependencies(captureStream, &node, 1, cudaStreamSetCaptureDependencies));

    // Insert kernel node D
    capturedWhileEmptyKernel<<<1, 1, 0, captureStream>>>();

    checkCudaErrors(cudaStreamEndCapture(captureStream, &graph));
    checkCudaErrors(cudaStreamDestroy(captureStream));

    // Populate conditional body graph using stream capture
    cudaStream_t bodyStream;
    checkCudaErrors(cudaStreamCreate(&bodyStream));

    checkCudaErrors(cudaStreamBeginCaptureToGraph(bodyStream, bodyGraph, nullptr, nullptr, 0, cudaStreamCaptureModeRelaxed));

    // Insert kernel node C
    capturedWhileKernel<<<1, 1, 0, bodyStream>>>(dPtr, handle);
    checkCudaErrors(cudaStreamEndCapture(bodyStream, nullptr));
    checkCudaErrors(cudaStreamDestroy(bodyStream));

    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Initialize device memory and launch the graph
    // Device memory is zero, so the conditional node will not execute
    checkCudaErrors(cudaMemset(dPtr, 0, 1)); // Set dPtr to 0
    printf("Host: Launching graph with loop counter set to 0\n");
    checkCudaErrors(cudaGraphLaunch(graphExec, 0));
    checkCudaErrors(cudaDeviceSynchronize());

    // Initialize device memory and launch the graph
    checkCudaErrors(cudaMemset(dPtr, 10, 1)); // Set dPtr to 10
    printf("Host: Launching graph with loop counter set to 10\n");
    checkCudaErrors(cudaGraphLaunch(graphExec, 0));
    checkCudaErrors(cudaDeviceSynchronize());

    // Cleanup
    checkCudaErrors(cudaGraphExecDestroy(graphExec));
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaFree(dPtr));

    printf("capturedWhileGraph: Complete\n\n");
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
    simpleDoWhileGraph();
    capturedWhileGraph();

    return 0;
}
