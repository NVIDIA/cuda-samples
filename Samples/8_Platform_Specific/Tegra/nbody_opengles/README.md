# nbody_opengles - CUDA N-Body Simulation with GLES

## Description

This sample demonstrates efficient all-pairs simulation of a gravitational n-body simulation in CUDA. Unlike the OpenGL nbody sample, there is no user interaction.

## Key Concepts

Graphics Interop, Data Parallel Algorithms, Physically-Based Simulation

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux

## Supported CPU Architecture

armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaGraphicsUnmapResources, cudaSetDeviceFlags, cudaGraphicsResourceSetMapFlags, cudaGraphicsResourceGetMappedPointer, cudaGraphicsMapResources, cudaSetDevice, cudaEventSynchronize, cudaGetDeviceProperties, cudaDeviceSynchronize, cudaEventRecord, cudaGetDevice, cudaMemcpyToSymbol, cudaStreamQuery, cudaEventDestroy, cudaEventElapsedTime, cudaGetDeviceCount, cudaEventCreate

## Dependencies needed to build/run
[X11](../../../README.md#x11), [GLES](../../../README.md#gles)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
