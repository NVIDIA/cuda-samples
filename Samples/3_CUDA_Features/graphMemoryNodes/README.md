# graphMemoryNodes - Graph Memory Nodes

## Description

A demonstration of memory allocations and frees within CUDA graphs using Graph APIs and Stream Capture APIs.

## Key Concepts

CUDA Graphs, Stream Capture

## Supported SM Architectures

[SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaMemcpy, cudaDeviceGetAttribute, cudaDriverGetVersion, cudaGraphLaunch, cudaEventDestroy, cudaMallocAsync, cudaStreamEndCapture, cudaMallocManaged, cudaGraphCreate, cudaMemcpyAsync, cudaFreeAsync, cudaStreamCreateWithFlags, cudaGraphInstantiate, cudaStreamDestroy, cudaStreamBeginCapture, cudaStreamWaitEvent, cudaEventCreate, cudaGraphAddMemAllocNode, cudaFree, cudaGraphAddKernelNode, cudaGraphAddMemFreeNode, cudaGraphDestroy, cudaEventRecord, cudaStreamSynchronize, cudaMalloc, cudaGraphExecDestroy

## Prerequisites

Download and install the [CUDA Toolkit 12.5](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## References (for more details)
