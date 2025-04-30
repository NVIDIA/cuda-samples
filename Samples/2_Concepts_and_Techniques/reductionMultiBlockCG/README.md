# reductionMultiBlockCG - Reduction using MultiBlock Cooperative Groups

## Description

This sample demonstrates single pass reduction using Multi Block Cooperative Groups.  This sample requires devices with compute capability 6.0 or higher having compute preemption.

## Key Concepts

Cooperative Groups, MultiBlock Cooperative Groups

## Supported SM Architectures

[SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, aarch64

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaMemcpy, cudaFree, cudaSetDevice, cudaDeviceSynchronize, cudaLaunchCooperativeKernel, cudaMalloc, cudaOccupancyMaxActiveBlocksPerMultiprocessor, cudaGetDeviceProperties, cudaOccupancyMaxPotentialBlockSize

## Dependencies needed to build/run
[MBCG](../../../README.md#mbcg), [CPP11](../../../README.md#cpp11)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
