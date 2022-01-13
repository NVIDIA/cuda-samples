# simpleD3D12 - Simple D3D12 CUDA Interop

## Description

A program which demonstrates Direct3D12 interoperability with CUDA.  The program creates a sinewave in DX12 vertex buffer which is created using CUDA kernels. DX12 and CUDA synchronizes using DirectX12 Fences. Direct3D then renders the results on the screen.  A DirectX12 Capable NVIDIA GPU is required on Windows10 or higher OS.

## Key Concepts

Graphics Interop, CUDA DX12 Interop, Image Processing

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Windows

## Supported CPU Architecture

x86_64

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaFree, cudaSignalExternalSemaphoresAsync, cudaStreamCreate, cudaGetDeviceCount, cudaImportExternalSemaphore, cudaGetDeviceProperties, cudaImportExternalMemory, cudaExternalMemoryGetMappedBuffer, cudaDestroyExternalSemaphore, cudaSetDevice, cudaWaitExternalSemaphoresAsync, cudaStreamSynchronize, cudaDestroyExternalMemory

## Dependencies needed to build/run
[DirectX12](../../README.md#directx12)

## Prerequisites

Download and install the [CUDA Toolkit 11.6](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run

### Windows
The Windows samples are built using the Visual Studio IDE. Solution files (.sln) are provided for each supported version of Visual Studio, using the format:
```
*_vs<version>.sln - for Visual Studio <version>
```
Each individual sample has its own set of solution files in its directory:

To build/examine all the samples at once, the complete solution files should be used. To build/examine a single sample, the individual sample solution files should be used.
> **Note:** Some samples require that the Microsoft DirectX SDK (June 2010 or newer) be installed and that the VC++ directory paths are properly set up (**Tools > Options...**). Check DirectX Dependencies section for details."

## References (for more details)

