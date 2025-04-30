# simpleVulkan - Vulkan CUDA Interop Sinewave

## Description

This sample demonstrates Vulkan CUDA Interop. CUDA imports the Vulkan vertex buffer and operates on it to create sinewave, and synchronizes with Vulkan through vulkan semaphores imported by CUDA. This sample depends on Vulkan SDK, GLFW3 libraries, for building this sample please refer to "Build_instructions.txt" provided in this sample's directory

## Key Concepts

Graphics Interop, CUDA Vulkan Interop, Data Parallel Algorithms

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, aarch64

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaStreamCreateWithFlags, cudaExternalMemoryGetMappedBuffer, cudaSignalSemaphore, cudaWaitExternalSemaphoresAsync, cudaVertMem, cudaImportExternalSemaphore, cudaWaitSemaphore, cudaHeightMap, cudaSetDevice, cudaGetDeviceCount, cudaSignalExternalSemaphoresAsync, cudaTimelineSemaphore, cudaStreamSynchronize, cudaDestroyExternalMemory, cudaOccupancyMaxActiveBlocksPerMultiprocessor, cudaImportExternalMemory, cudaGetDeviceProperties, cudaDestroyExternalSemaphore

## Dependencies needed to build/run
[X11](../../../README.md#x11), [VULKAN](../../../README.md#vulkan)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)
