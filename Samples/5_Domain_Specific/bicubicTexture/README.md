# bicubicTexture - Bicubic B-spline Interoplation

## Description

This sample demonstrates how to efficiently implement a Bicubic B-spline interpolation filter with CUDA texture.

## Key Concepts

Graphics Interop, Image Processing

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaGraphicsUnmapResources, cudaCreateChannelDesc, cudaMallocArray, cudaFreeArray, cudaFree, cudaMemcpy, cudaGraphicsResourceGetMappedPointer, cudaGraphicsMapResources, cudaDestroyTextureObject, cudaDeviceSynchronize, cudaCreateTextureObject, cudaGraphicsUnregisterResource, cudaMalloc, cudaGraphicsGLRegisterBuffer, cudaGetDeviceProperties

## Dependencies needed to build/run
[X11](../../../README.md#x11), [GL](../../../README.md#gl)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
