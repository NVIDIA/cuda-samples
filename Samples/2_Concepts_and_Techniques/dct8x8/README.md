# dct8x8 - DCT8x8

## Description

This sample demonstrates how Discrete Cosine Transform (DCT) for blocks of 8 by 8 pixels can be performed using CUDA: a naive implementation by definition and a more traditional approach used in many libraries. As opposed to implementing DCT in a fragment shader, CUDA allows for an easier and more efficient implementation.

## Key Concepts

Image Processing, Video Compression

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaMallocArray, cudaFreeArray, cudaFree, cudaMallocPitch, cudaDestroyTextureObject, cudaDeviceSynchronize, cudaCreateTextureObject

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## References (for more details)

[whitepaper](./doc/dct8x8.pdf)
