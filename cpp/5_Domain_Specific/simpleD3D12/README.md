# simpleD3D12 - Simple D3D12 CUDA Interop

## Description

A program which demonstrates Direct3D12 interoperability with CUDA.  The program creates a sinewave in DX12 vertex buffer which is created using CUDA kernels. DX12 and CUDA synchronizes using DirectX12 Fences. Direct3D then renders the results on the screen.  A DirectX12 Capable NVIDIA GPU is required on Windows10 or higher OS.

## Key Concepts

Graphics Interop, CUDA DX12 Interop, Image Processing

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Windows

## Supported CPU Architecture

x86_64

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaWaitExternalSemaphoresAsync, cudaExternalMemoryGetMappedBuffer, cudaImportExternalSemaphore, cudaFree, cudaSetDevice, cudaSignalExternalSemaphoresAsync, cudaGetDeviceProperties, cudaStreamSynchronize, cudaDestroyExternalMemory, cudaStreamCreate, cudaImportExternalMemory, cudaGetDeviceCount, cudaDestroyExternalSemaphore

## Dependencies needed to build/run
[DirectX12](../../../README.md#directx12)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
