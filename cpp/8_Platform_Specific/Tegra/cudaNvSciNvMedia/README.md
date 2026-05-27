# cudaNvSciNvMedia - NvMedia CUDA Interop

## Description

This sample demonstrates CUDA-NvMedia interop via NvSciBuf/NvSciSync APIs. Note that this sample only supports cross build from x86_64 to aarch64, aarch64 native build is not supported. For detailed workflow of the sample please check cudaNvSciNvMedia_Readme.pdf in the sample directory.

## Key Concepts

CUDA NvSci Interop, Data Parallel Algorithms, Image Processing

## Supported SM Architectures

[SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, QNX

## Supported CPU Architecture

aarch64

## CUDA APIs involved

### [CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
cuDeviceGetUuid

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaImportExternalSemaphore, cudaGetMipmappedArrayLevel, cudaSetDevice, cudaDestroySurfaceObject, cudaCreateSurfaceObject, cudaImportNvSciImage, cudaCreateChannelDesc, cudaMallocHost, cudaSignalExternalSemaphoresAsync, cudaFreeHost, cudaMemcpyAsync, cudaStreamCreateWithFlags, cudaExternalMemoryGetMappedMipmappedArray, cudaMallocArray, cudaFreeArray, cudaStreamDestroy, cudaDeviceGetNvSciSyncAttributes, cudaDestroyExternalMemory, cudaImportExternalMemory, cudaDestroyExternalSemaphore, cudaFreeMipmappedArray, cudaImportNvSciSync, cudaFree, cudaStreamSynchronize, cudaMalloc, cudaWaitExternalSemaphoresAsync

## Dependencies needed to build/run
[NVSCI](../../../README.md#nvsci), [NvMedia](../../../README.md#nvmedia)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
