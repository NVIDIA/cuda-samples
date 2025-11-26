# asyncAPI - asyncAPI

## Description

This sample illustrates the usage of CUDA events for both GPU timing and overlapping CPU and GPU execution. Events are inserted into a stream of CUDA calls. Since CUDA stream calls are asynchronous, the CPU can perform computations while GPU is executing (including DMA memcopies between the host and device). CPU can query CUDA events to determine whether GPU has completed tasks.

## Key Concepts

Asynchronous Data Transfers, CUDA Streams and Events

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaProfilerStop, cudaMalloc, cudaMemcpyAsync, cudaFree, cudaMallocHost, cudaProfilerStart, cudaDeviceSynchronize, cudaEventRecord, cudaFreeHost, cudaMemset, cudaEventDestroy, cudaEventQuery, cudaEventElapsedTime, cudaGetDeviceProperties, cudaEventCreate

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## References (for more details)
