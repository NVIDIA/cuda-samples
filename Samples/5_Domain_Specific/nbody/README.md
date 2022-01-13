# nbody - CUDA N-Body Simulation

## Description

This sample demonstrates efficient all-pairs simulation of a gravitational n-body simulation in CUDA.  This sample accompanies the GPU Gems 3 chapter "Fast N-Body Simulation with CUDA".  With CUDA 5.5, performance on Tesla K20c has increased to over 1.8TFLOP/s single precision.  Double Performance has also improved on all Kepler and Fermi GPU architectures as well.  Starting in CUDA 4.0, the nBody sample has been updated to take advantage of new features to easily scale the n-body simulation across multiple GPUs in a single PC.  Adding "-numbodies=<bodies>" to the command line will allow users to set # of bodies for simulation.  Adding “-numdevices=<N>” to the command line option will cause the sample to use N devices (if available) for simulation.  In this mode, the position and velocity data for all bodies are read from system memory using “zero copy” rather than from device memory.  For a small number of devices (4 or fewer) and a large enough number of bodies, bandwidth is not a bottleneck so we can achieve strong scaling across these devices.

## Key Concepts

Graphics Interop, Data Parallel Algorithms, Physically-Based Simulation

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaMemcpyToSymbol, cudaGraphicsMapResources, cudaEventRecord, cudaStreamQuery, cudaEventCreate, cudaGraphicsResourceGetMappedPointer, cudaGetDeviceCount, cudaEventElapsedTime, cudaDeviceSynchronize, cudaEventSynchronize, cudaGraphicsResourceSetMapFlags, cudaSetDeviceFlags, cudaEventDestroy, cudaDeviceCanAccessPeer, cudaSetDevice, cudaGraphicsUnmapResources, cudaGetDeviceProperties, cudaGetDevice

## Dependencies needed to build/run
[X11](../../README.md#x11), [GL](../../README.md#gl)

## Prerequisites

Download and install the [CUDA Toolkit 11.6](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run

### Windows
The Windows samples are built using the Visual Studio IDE. Solution files (.sln) are provided for each supported version of Visual Studio, using the format:
```
*_vs<version>.sln - for Visual Studio <version>
```
Each individual sample has its own set of solution files in its directory:

To build/examine all the samples at once, the complete solution files should be used. To build/examine a single sample, the individual sample solution files should be used.
> **Note:** Some samples require that the Microsoft DirectX SDK (June 2010 or newer) be installed and that the VC++ directory paths are properly set up (**Tools > Options...**). Check DirectX Dependencies section for details."

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
```
The samples makefiles can take advantage of certain options:
*  **TARGET_ARCH=<arch>** - cross-compile targeting a specific architecture. Allowed architectures are x86_64, armv7l.
    By default, TARGET_ARCH is set to HOST_ARCH. On a x86_64 machine, not setting TARGET_ARCH is the equivalent of setting TARGET_ARCH=x86_64.<br/>
`$ make TARGET_ARCH=x86_64` <br/> `$ make TARGET_ARCH=armv7l` <br/>
    See [here](http://docs.nvidia.com/cuda/cuda-samples/index.html#cross-samples) for more details.
*   **dbg=1** - build with debug symbols
    ```
    $ make dbg=1
    ```
*   **SMS="A B ..."** - override the SM architectures for which the sample will be built, where `"A B ..."` is a space-delimited list of SM architectures. For example, to generate SASS for SM 50 and SM 60, use `SMS="50 60"`.
    ```
    $ make SMS="50 60"
    ```

*  **HOST_COMPILER=<host_compiler>** - override the default g++ host compiler. See the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements) for a list of supported host compilers.
```
    $ make HOST_COMPILER=g++
```

## References (for more details)

[whitepaper](./doc/nbody_gems3_ch31.pdf)
