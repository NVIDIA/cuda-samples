# bindlessTexture - Bindless Texture

## Description

This example demonstrates use of cudaSurfaceObject, cudaTextureObject, and MipMap support in CUDA.  A GPU with Compute Capability SM 3.0 is required to run the sample.

## Key Concepts

Graphics Interop, Texture

## Supported SM Architectures

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaMemcpy, cudaGetMipmappedArrayLevel, cudaGraphicsMapResources, cudaDestroySurfaceObject, cudaExtent, cudaDeviceSynchronize, cudaCreateSurfaceObject, cudaMallocMipmappedArray, cudaPitchedPtr, cudaGraphicsResourceGetMappedPointer, cudaCreateTextureObject, cudaGraphicsUnmapResources, cudaMallocArray, cudaFreeArray, cudaArrayGetInfo, cudaGetLastError, cudaDestroyTextureObject, cudaGraphicsGLRegisterBuffer, cudaFreeMipmappedArray, cudaFree, cudaGraphicsUnregisterResource, cudaMalloc

## Dependencies needed to build/run
[X11](../../../README.md#x11), [GL](../../../README.md#gl)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
