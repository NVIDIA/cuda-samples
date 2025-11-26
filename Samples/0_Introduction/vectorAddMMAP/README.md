# vectorAddMMAP - Vector Addition cuMemMap

## Description

This sample replaces the device allocation in the vectorAddDrv sample with cuMemMap-ed allocations.  This sample demonstrates that the cuMemMap api allows the user to specify the physical properties of their memory while retaining the contiguous nature of their access, thus not requiring a change in their program structure.

## Key Concepts

CUDA Driver API, Vector Addition, MMAP

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, ppc64le

## CUDA APIs involved

### [CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
cuMemcpyDtoH, cuDeviceCanAccessPeer, cuModuleGetFunction, cuMemSetAccess, cuMemRelease, cuInit, cuMemcpyHtoD, cuLaunchKernel, cuMemCreate, cuModuleLoadData, cuCtxDestroy, cuDeviceGetCount, cuMemMap, cuDeviceGetAttribute, cuMemGetAllocationGranularity, cuMemAddressFree, cuMemUnmap, cuCtxCreate, cuMemAddressReserve

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## References (for more details)
