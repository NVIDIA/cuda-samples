# graphMemoryFootprint - Graph Memory Footprint

## Description

This sample demonstrates how graph memory nodes re-use virtual addresses and physical memory.

## Key Concepts

CUDA Runtime API, Performance Strategies, CUDA Graphs

## Supported SM Architectures

[SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaGraphAddMemAllocNode, cudaStreamCreateWithFlags, cudaGraphInstantiate, cudaStreamDestroy, cudaFree, cudaDeviceGetAttribute, cudaGraphAddKernelNode, cudaGraphAddMemFreeNode, cudaDeviceGetGraphMemAttribute, cudaGraphCreate, cudaGraphDestroy, cudaDriverGetVersion, cudaGraphLaunch, cudaStreamSynchronize, cudaDeviceGraphMemTrim, cudaGetDeviceProperties, cudaGraphExecDestroy

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## References (for more details)
