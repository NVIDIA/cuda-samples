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
 * Demonstrates CUDA Graph construction via stream capture.
 *
 * Operations are issued to a stream between cudaStreamBeginCapture and
 * cudaStreamEndCapture. Instead of executing, they are recorded and the
 * runtime automatically builds this graph:
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

#define GRAPH_LAUNCH_ITERATIONS 3

void cudaGraphsUsingStreamCapture(float  *inputVec_h,
                                  float  *inputVec_d,
                                  double *outputVec_d,
                                  double *result_d,
                                  size_t  inputSize,
                                  size_t  numOfBlocks)
{
    cudaStream_t stream1, streamForGraph;
    cudaGraph_t  graph;
    double       result_h = 0.0;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&streamForGraph);

    // Begin capture: operations issued to stream1 are recorded, not executed.
    cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

    cudaMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize, cudaMemcpyDefault, stream1);

    reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream1>>>(inputVec_d, outputVec_d, inputSize, numOfBlocks);

    reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_d, result_d, numOfBlocks);
    cudaMemcpyAsync(&result_h, result_d, sizeof(double), cudaMemcpyDefault, stream1);

    callBackData_t hostFnData = {0};
    hostFnData.data           = &result_h;
    hostFnData.fn_name        = "cudaGraphsUsingStreamCapture";
    cudaLaunchHostFunc(stream1, myHostNodeCallback, &hostFnData);

    // End capture: the runtime builds a graph from everything recorded above.
    // The resulting graph is structurally identical to the one built manually.
    cudaStreamEndCapture(stream1, &graph);

    size_t numNodes = 0;
    cudaGraphGetNodes(graph, NULL, &numNodes);
    printf("Graph node count: %zu\n", numNodes);

    // Instantiate the captured graph into an executable form.
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
    cudaStreamDestroy(stream1);
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

    printf("=== Stream Capture ===\n");
    cudaGraphsUsingStreamCapture(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);

    cudaFree(inputVec_d);
    cudaFree(outputVec_d);
    cudaFree(result_d);
    cudaFreeHost(inputVec_h);
    return EXIT_SUCCESS;
}
