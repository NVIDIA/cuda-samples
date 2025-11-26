# conjugateGradientCudaGraphs - Conjugate Gradient using Cuda Graphs

## Description

This sample implements a conjugate gradient solver on GPU using CUBLAS and CUSPARSE library calls captured and called using CUDA Graph APIs.

## Key Concepts

Linear Algebra, CUBLAS Library, CUSPARSE Library

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaGraphInstantiate, cudaStreamDestroy, cudaStreamBeginCapture, cudaFree, cudaMallocHost, cudaStreamEndCapture, cudaGraphDestroy, cudaFreeHost, cudaGraphLaunch, cudaStreamCreate, cudaStreamSynchronize, cudaOccupancyMaxPotentialBlockSize, cudaMalloc, cudaMemcpyAsync, cudaMemsetAsync, cudaGetDeviceProperties, cudaGraphExecDestroy

## Dependencies needed to build/run
[CUBLAS](../../../README.md#cublas), [CUSPARSE](../../../README.md#cusparse)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
