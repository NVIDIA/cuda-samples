# EGLStream_CUDA_CrossGPU - EGLStream_CUDA_CrossGPU

## Description

Demonstrates CUDA and EGL Streams interop, where consumer's EGL Stream is on one GPU and producer's on other and both consumer-producer are different processes.

## Key Concepts

EGLStreams Interop

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
cuDeviceGetName, cuEGLStreamConsumerReleaseFrame, cuEGLStreamConsumerConnect, cuEGLStreamConsumerDisconnect, cuCtxPushCurrent, cuEGLStreamProducerReturnFrame, cuStreamCreate, cuEGLStreamProducerPresentFrame, cuMemFree, cuGraphicsResourceGetMappedEglFrame, cuInit, cuMemcpyHtoD, cuDeviceGet, cuEGLStreamConsumerAcquireFrame, cuEGLStreamProducerDisconnect, cuEGLStreamProducerConnect, cuDeviceGetAttribute, cuCtxSynchronize, cuMemAlloc, cuCtxPopCurrent, cuCtxCreate

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaMemcpy, cudaMalloc, cudaProducerPresentFrame, cudaFree, cudaGetErrorString, cudaConsumerReleaseFrame, cudaProducerReturnFrame, cudaDeviceSynchronize, cudaDeviceCreateProducer, cudaProducerDeinit, cudaProducerPrepareFrame, cudaGetValueMismatch, cudaConsumerAcquireFrame, cudaProducerInit, cudaDeviceCreateConsumer

## Dependencies needed to build/run
[EGL](../../../README.md#egl)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
