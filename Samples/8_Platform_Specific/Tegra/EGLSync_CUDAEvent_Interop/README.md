# EGLSync_CUDAEvent_Interop - EGLSync CUDA Event Interop

## Description

Demonstrates interoperability between CUDA Event and EGL Sync/EGL Image using which one can achieve synchronization on GPU itself for GL-EGL-CUDA operations instead of blocking CPU for synchronization.

## Key Concepts

EGLSync-CUDAEvent Interop, EGLImage-CUDA Interop

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux

## Supported CPU Architecture

armv7l

## CUDA APIs involved

### [CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
cuEventRecord, cuDeviceGetAttribute, cuEventCreate, cuCtxSynchronize, cuEventDestroy, cuGraphicsEGLRegisterImage, cuGraphicsSubResourceGetMappedArray, cuStreamCreate, cuStreamWaitEvent, cuGraphicsUnregisterResource, cuCtxCreate, cuSurfObjectCreate, cuEventCreateFromEGLSync, cuCtxPushCurrent, cuInit

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaMemcpy, cudaGetErrorString, cudaFree, cudaDeviceSynchronize, cudaGetValueMismatch, cudaMalloc

## Dependencies needed to build/run
[EGL](../../../README.md#egl), [EGLSync](../../../README.md#eglsync), [X11](../../../README.md#x11), [GLES](../../../README.md#gles)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
