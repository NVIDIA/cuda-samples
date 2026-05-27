# jacobiCudaGraphs - Jacobi CUDA Graphs

## Description

Demonstrates Instantiated CUDA Graph Update with Jacobi Iterative Method using cudaGraphExecKernelNodeSetParams() and cudaGraphExecUpdate() approach.

## Key Concepts

CUDA Graphs, Stream Capture, Instantiated CUDA Graph Update, Cooperative Groups

## Supported SM Architectures

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaExtent, cudaGraphLaunch, cudaGraphAddMemcpyNode, cudaMallocHost, cudaPitchedPtr, cudaStreamEndCapture, cudaGraphCreate, cudaFreeHost, cudaMemsetAsync, cudaMemcpyAsync, cudaGraphExecKernelNodeSetParams, cudaStreamCreateWithFlags, cudaGraphInstantiate, cudaStreamBeginCapture, cudaFree, cudaGraphExecUpdate, cudaGraphAddKernelNode, cudaPos, cudaStreamSynchronize, cudaGraphAddMemsetNode, cudaMalloc, cudaGraphExecDestroy

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## References (for more details)
