# fluidsGLES - Fluids (OpenGLES Version)

## Description

An example of fluid simulation using CUDA and CUFFT, with OpenGLES rendering.

## Key Concepts

Graphics Interop, CUFFT Library, Physically-Based Simulation

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux

## Supported CPU Architecture

armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaGraphicsUnmapResources, cudaMemcpy, cudaMallocArray, cudaFreeArray, cudaFree, cudaMallocPitch, cudaGraphicsResourceGetMappedPointer, cudaGraphicsMapResources, cudaDestroyTextureObject, cudaCreateTextureObject, cudaGraphicsUnregisterResource, cudaMalloc, cudaGraphicsGLRegisterBuffer, cudaGetDeviceProperties

## Dependencies needed to build/run
[X11](../../../README.md#x11), [GLES](../../../README.md#gles), [CUFFT](../../../README.md#cufft)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
