# threadFenceReduction - threadFenceReduction

## Description

This sample shows how to perform a reduction operation on an array of values using the thread Fence intrinsic to produce a single value in a single kernel (as opposed to two or more kernel calls as shown in the "reduction" CUDA Sample).  Single-pass reduction requires global atomic instructions (Compute Capability 2.0 or later) and the _threadfence() intrinsic (CUDA 2.2 or later).

## Key Concepts

Cooperative Groups, Data-Parallel Algorithms, Performance Strategies

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaMemcpy, cudaFree, cudaDeviceSynchronize, cudaMalloc, cudaGetDeviceProperties

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## References (for more details)
