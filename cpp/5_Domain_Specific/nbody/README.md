# nbody - CUDA N-Body Simulation

## Description

This sample demonstrates efficient all-pairs simulation of a gravitational n-body simulation in CUDA.  This sample accompanies the GPU Gems 3 chapter "Fast N-Body Simulation with CUDA".  With CUDA 5.5, performance on Tesla K20c has increased to over 1.8TFLOP/s single precision.  Double Performance has also improved on all Kepler and Fermi GPU architectures as well.  Starting in CUDA 4.0, the nBody sample has been updated to take advantage of new features to easily scale the n-body simulation across multiple GPUs in a single PC.  Adding "-numbodies=<bodies>" to the command line will allow users to set # of bodies for simulation.  Adding “-numdevices=<N>” to the command line option will cause the sample to use N devices (if available) for simulation.  In this mode, GPU 0 holds the authoritative position and velocity buffers.  Each other device keeps a local scratch copy of the read-only position array and refreshes it with one peer-to-peer copy per simulation step, so the all-pairs kernel reads every tile from local memory instead of re-fetching it over the peer link on every tile access; each device still writes its own share of results directly back to GPU 0 over the peer link.  Each device issues that copy and its kernel on its own non-blocking stream so the GPUs overlap.  This keeps per-step peer traffic proportional to the number of devices instead of the number of thread blocks, so multi-GPU runs scale on both NVLink and PCIe-only (no NVLink) systems.

## Key Concepts

Graphics Interop, Data Parallel Algorithms, Physically-Based Simulation

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaGraphicsUnmapResources, cudaSetDeviceFlags, cudaGraphicsResourceSetMapFlags, cudaGraphicsResourceGetMappedPointer, cudaGraphicsMapResources, cudaSetDevice, cudaEventSynchronize, cudaGetDeviceCount, cudaGetDeviceProperties, cudaDeviceSynchronize, cudaEventRecord, cudaGetDevice, cudaMemcpyToSymbol, cudaEventDestroy, cudaEventElapsedTime, cudaDeviceCanAccessPeer, cudaDeviceEnablePeerAccess, cudaMemcpyPeerAsync, cudaStreamCreateWithFlags, cudaStreamDestroy, cudaStreamSynchronize, cudaEventCreate

## Dependencies needed to build/run
[X11](../../../README.md#x11), [OpenGL](../../../README.md#opengl), [Freeglut](../../../README.md#freeglut), [GLEW](../../../README.md#glew)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)

[whitepaper](./doc/nbody_gems3_ch31.pdf)
