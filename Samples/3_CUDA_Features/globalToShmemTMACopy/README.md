# globalToShmemTMACopy - Global Memory to Shared Memory TMA Copy

## Description

This sample shows how to use the CUDA driver API and inline PTX assembly to copy
a 2D tile of a tensor into shared memory. It also demonstrates arrive-wait
barrier for synchronization. 

## Key Concepts

CUDA Runtime API, CUDA Driver API, PTX ISA, CPP11 CUDA

## Supported SM Architectures

This sample requires compute capability 9.0 or higher.

[SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows, QNX

## Supported CPU Architecture

x86_64, ppc64le, armv7l, aarch64

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaMalloc, cudaMemcpy, cudaFree, cudaDeviceSynchronize

### [CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
cudaMalloc, cudaMemcpy, cudaFree, cudaDeviceSynchronize

### [CUDA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

## Dependencies needed to build/run
[CPP11](../../../README.md#cpp11)

## Prerequisites

Download and install the [CUDA Toolkit 12.2](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run


## References (for more details)

