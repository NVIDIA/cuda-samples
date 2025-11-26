# simpleIPC - simpleIPC

## Description

This CUDA Runtime API sample is a very basic sample that demonstrates Inter Process Communication with one process per GPU for computation.  Requires Compute Capability 3.0 or higher and a Linux Operating System, or a Windows Operating System.

## Key Concepts

CUDA Systems Integration, Peer to Peer, InterProcess Communication

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, ppc64le

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaSetDevice, cudaIpcCloseMemHandle, cudaEventDestroy, cudaGetDeviceCount, cudaMemcpyAsync, cudaDeviceCanAccessPeer, cudaStreamCreateWithFlags, cudaStreamDestroy, cudaGetLastError, cudaIpcOpenEventHandle, cudaIpcOpenMemHandle, cudaIpcGetEventHandle, cudaStreamWaitEvent, cudaEventCreate, cudaFree, cudaEventSynchronize, cudaEventRecord, cudaIpcGetMemHandle, cudaStreamSynchronize, cudaDeviceEnablePeerAccess, cudaMalloc, cudaOccupancyMaxActiveBlocksPerMultiprocessor, cudaGetDeviceProperties

## Dependencies needed to build/run
[IPC](../../../README.md#ipc)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
