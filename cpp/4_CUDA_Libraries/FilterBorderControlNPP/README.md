# FilterBorderControlNPP - Filter Border Control NPP

## Description

This sample demonstrates how any border version of an NPP filtering function can be used in the most common mode, with border control enabled. Mentioned functions can be used to duplicate the results of the equivalent non-border version of the NPP functions. They can be also used for enabling and disabling border control on various source image edges depending on what portion of the source image is being used as input.

## Key Concepts

Performance Strategies, Image Processing, NPP Library

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaRuntimeGetVersion, cudaDeviceReset, cudaSetDevice, cudaGetDeviceCount, cudaDeviceInit, cudaDriverGetVersion, cudaGetDeviceProperties

## Dependencies needed to build/run
[FreeImage](../../../README.md#freeimage), [NPP](../../../README.md#npp)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
