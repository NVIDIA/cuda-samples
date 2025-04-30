# conjugateGradientMultiDeviceCG - conjugateGradient using MultiDevice Cooperative Groups

## Description

This sample implements a conjugate gradient solver on multiple GPUs using Multi Device Cooperative Groups, also uses Unified Memory optimized using prefetching and usage hints.

## Key Concepts

Unified Memory, Linear Algebra, Cooperative Groups, MultiDevice Cooperative Groups, CUBLAS Library, CUSPARSE Library

## Supported SM Architectures

[SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, aarch64

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaHostAlloc, cudaMemPrefetchAsync, cudaFree, cudaLaunchCooperativeKernel, cudaMallocManaged, cudaSetDevice, cudaGetDeviceCount, cudaGetDeviceProperties, cudaFreeHost, cudaMemset, cudaStreamCreate, cudaStreamSynchronize, cudaDeviceEnablePeerAccess, cudaMemAdvise, cudaOccupancyMaxActiveBlocksPerMultiprocessor, cudaDeviceCanAccessPeer

## Dependencies needed to build/run
[UVM](../../../README.md#uvm), [MDCG](../../../README.md#mdcg), [CPP11](../../../README.md#cpp11)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
