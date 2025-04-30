# clock_nvrtc - Clock libNVRTC

## Description

This example shows how to use the clock function using libNVRTC to measure the performance of block of threads of a kernel accurately.

## Key Concepts

Performance Strategies, Runtime Compilation

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows, QNX

## Supported CPU Architecture

x86_64, aarch64

## CUDA APIs involved

### [CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
cuMemcpyDtoH, cuLaunchKernel, cuMemcpyHtoD, cuCtxSynchronize, cuMemAlloc, cuMemFree, cuModuleGetFunction

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaBlockSize, cudaGridSize

## Dependencies needed to build/run
[NVRTC](../../../README.md#nvrtc)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
