# simpleD3D11 - Simple D3D11

## Description

Simple program which demonstrates  how to use the CUDA D3D11 External Resource Interoperability APIs to update D3D11 buffers from CUDA and synchronize between D3D11 and CUDA with Keyed Mutexes.

## Key Concepts

Graphics Interop, Image Processing

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Windows

## Supported CPU Architecture

x86_64

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaImportKeyedMutex, cudaExternalMemoryGetMappedBuffer, cudaStreamCreateWithFlags, cudaWaitExternalSemaphoresAsync, cudaImportExternalSemaphore, cudaFree, cudaImportVertexBuffer, cudaReleaseSync, cudaSetDevice, cudaSignalExternalSemaphoresAsync, cudaAcquireSync, cudaDestroyExternalMemory, cudaImportExternalMemory, cudaGetDeviceCount, cudaDestroyExternalSemaphore

## Dependencies needed to build/run
[DirectX](../../../README.md#directx)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
