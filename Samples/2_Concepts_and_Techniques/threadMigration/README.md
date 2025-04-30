# threadMigration - CUDA Context Thread Management

## Description

Simple program illustrating how to the CUDA Context Management API and uses the new CUDA 4.0 parameter passing and CUDA launch API.  CUDA contexts can be created separately and attached independently to different threads.

## Key Concepts

CUDA Driver API

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
cuMemcpyDtoH, cuLaunchKernel, cuModuleLoadData, cuDeviceGetName, cuDeviceGet, cuDeviceGetAttribute, cuMemAlloc, cuMemFree, cuCtxDestroy, cuCtxPopCurrent, cuModuleUnload, cuDeviceGetCount, cuModuleGetFunction, cuCtxCreate, cuCtxPushCurrent, cuInit

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## References (for more details)
