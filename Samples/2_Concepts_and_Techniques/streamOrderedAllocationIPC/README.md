# streamOrderedAllocationIPC - stream Ordered Allocation IPC Pools

## Description

This sample demonstrates IPC pools of stream ordered memory allocated using cudaMallocAsync and cudaMemPool family of APIs.

## Key Concepts

Performance Strategies

## Supported SM Architectures

[SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux

## Supported CPU Architecture

x86_64

## CUDA APIs involved

### [CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
cuDeviceGetAttribute, cuDeviceGet

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaDeviceGetAttribute, cudaMemPoolImportFromShareableHandle, cudaSetDevice, cudaMemPoolExportPointer, cudaMemPoolGetAccess, cudaMemPoolDestroy, cudaMemPoolSetAccess, cudaMallocAsync, cudaMemPoolImportPointer, cudaGetDeviceCount, cudaMemcpyAsync, cudaDeviceCanAccessPeer, cudaFreeAsync, cudaStreamCreateWithFlags, cudaStreamDestroy, cudaGetLastError, cudaMemPoolCreate, cudaMemPoolExportToShareableHandle, cudaStreamSynchronize, cudaDeviceEnablePeerAccess, cudaOccupancyMaxActiveBlocksPerMultiprocessor, cudaGetDeviceProperties

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## References (for more details)
