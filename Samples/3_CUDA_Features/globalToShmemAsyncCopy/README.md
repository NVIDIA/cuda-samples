# globalToShmemAsyncCopy - Global Memory to Shared Memory Async Copy

## Description

This sample implements matrix multiplication which uses asynchronous copy of data from global to shared memory when on compute capability 8.0 or higher. Also demonstrates arrive-wait barrier for synchronization.

## Key Concepts

CUDA Runtime API, Linear Algebra, CPP11 CUDA

## Supported SM Architectures

[SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows, QNX

## Supported CPU Architecture

x86_64, armv7l, aarch64

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaStreamCreateWithFlags, cudaMalloc, cudaDeviceGetAttribute, cudaFree, cudaMallocHost, cudaEventSynchronize, cudaEventRecord, cudaFreeHost, cudaStreamSynchronize, cudaEventDestroy, cudaEventElapsedTime, cudaMemsetAsync, cudaMemcpyAsync, cudaEventCreate

## Dependencies needed to build/run
[CPP11](../../../README.md#cpp11)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
