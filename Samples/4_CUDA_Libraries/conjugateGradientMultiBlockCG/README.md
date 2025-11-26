# conjugateGradientMultiBlockCG - conjugateGradient using MultiBlock Cooperative Groups

## Description

This sample implements a conjugate gradient solver on GPU using Multi Block Cooperative Groups, also uses Unified Memory.

## Key Concepts

Unified Memory, Linear Algebra, Cooperative Groups, MultiBlock Cooperative Groups, CUBLAS Library, CUSPARSE Library

## Supported SM Architectures

[SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, aarch64

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaFree, cudaMallocManaged, cudaDeviceSynchronize, cudaEventRecord, cudaLaunchCooperativeKernel, cudaEventDestroy, cudaEventElapsedTime, cudaOccupancyMaxActiveBlocksPerMultiprocessor, cudaGetDeviceProperties, cudaEventCreate

## Dependencies needed to build/run
[UVM](../../../README.md#uvm), [MBCG](../../../README.md#mbcg)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
