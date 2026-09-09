# cudaNvSci - CUDA NvSciBuf/NvSciSync Interop

## Description

This sample demonstrates CUDA-NvSciBuf/NvSciSync Interop. Two CPU threads import the NvSciBuf and NvSciSync into CUDA to perform two image processing algorithms on a ppm image - image rotation in 1st thread &amp; rgba to grayscale conversion of rotated image in 2nd thread.

## Key Concepts

CUDA NvSci Interop, Data Parallel Algorithms, Image Processing

## Supported SM Architectures

[SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, QNX (standard and safety cross-builds)

## Supported CPU Architecture

x86_64, aarch64

## CUDA APIs involved

### [CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
cuDeviceGetUuid

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaExternalMemoryGetMappedBuffer, cudaImportExternalSemaphore, cudaDeviceGetAttribute, cudaNvSciSignal, cudaGetMipmappedArrayLevel, cudaImportNvSciRawBuf, cudaSetDevice, cudaImportNvSciImage, cudaNvSciApp, cudaDeviceId, cudaMallocHost, cudaSignalExternalSemaphoresAsync, cudaCreateTextureObject, cudaFreeHost, cudaNvSci, cudaNvSciWait, cudaGetDeviceCount, cudaMemcpyAsync, cudaStreamCreateWithFlags, cudaExternalMemoryGetMappedMipmappedArray, cudaStreamDestroy, cudaDeviceGetNvSciSyncAttributes, cudaDestroyTextureObject, cudaDestroyExternalMemory, cudaImportExternalMemory, cudaDestroyExternalSemaphore, cudaFreeMipmappedArray, cudaFree, cudaStreamSynchronize, cudaWaitExternalSemaphoresAsync, cudaImportNvSciSemaphore

## Dependencies needed to build/run
[NVSCI](../../../README.md#nvsci)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

On a QNX cross-build the NvSci headers and libraries come from the target filesystem, so pass `-DTARGET_FS=/path/to/qnx/targetfs` to cmake. Without it the build reports `NvSCI not found` and skips this sample. See the [QNX build instructions](../../../README.md#qnx).

## References (for more details)
