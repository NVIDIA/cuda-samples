# UnifiedMemoryStreams - Unified Memory Streams

## Description

This sample demonstrates the use of OpenMP and streams with Unified Memory on a single GPU.

## Key Concepts

CUDA Systems Integration, OpenMP, CUBLAS, Multithreading, Unified Memory, CUDA Streams and Events

## Supported SM Architectures

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaStreamDestroy, cudaFree, cudaMallocManaged, cudaStreamAttachMemAsync, cudaSetDevice, cudaDeviceSynchronize, cudaStreamSynchronize, cudaStreamCreate, cudaGetDeviceProperties

## Dependencies needed to build/run
[OpenMP](../../../README.md#openmp), [UVM](../../../README.md#uvm), [CUBLAS](../../../README.md#cublas)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
