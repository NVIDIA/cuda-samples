# CUDA Samples

Samples for CUDA Developers which demonstrates features in CUDA Toolkit. This version supports [CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-downloads).

## Release Notes

This section describes the release notes for the CUDA Samples on GitHub only.

### CUDA 11.7
* Added new folder structure for samples
* Added support of Visual Studio 2022 to all samples supported on [Windows](#windows-1).
* All CUDA samples are now only available on [GitHub](https://github.com/nvidia/cuda-samples). They are no longer available via CUDA toolkit.

### [older versions...](./CHANGELOG.md)

## Getting Started

### Prerequisites

Download and install the [CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
For system requirements and installation instructions of cuda toolkit, please refer to the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/), and the [Windows Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

### Getting the CUDA Samples

Using git clone the repository of CUDA Samples using the command below.
```
git clone https://github.com/NVIDIA/cuda-samples.git
```

Without using git the easiest way to use these samples is to download the zip file containing the current version by clicking the "Download ZIP" button on the repo page. You can then unzip the entire archive and use the samples.

## Building CUDA Samples

### Windows

The Windows samples are built using the Visual Studio IDE. Solution files (.sln) are provided for each supported version of Visual Studio, using the format:
```
*_vs<version>.sln - for Visual Studio <version>
```
Complete samples solution files exist at parent directory of the repo:

Each individual sample has its own set of solution files at:
`<CUDA_SAMPLES_REPO>\Samples\<sample_dir>\`

To build/examine all the samples at once, the complete solution files should be used. To build/examine a single sample, the individual sample solution files should be used.

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
```
The samples makefiles can take advantage of certain options:
*  **TARGET_ARCH=<arch>** - cross-compile targeting a specific architecture. Allowed architectures are x86_64, ppc64le, armv7l, aarch64.
    By default, TARGET_ARCH is set to HOST_ARCH. On a x86_64 machine, not setting TARGET_ARCH is the equivalent of setting TARGET_ARCH=x86_64.<br/>
`$ make TARGET_ARCH=x86_64` <br/> `$ make TARGET_ARCH=ppc64le` <br/> `$ make TARGET_ARCH=armv7l` <br/> `$ make TARGET_ARCH=aarch64` <br/>
    See [here](http://docs.nvidia.com/cuda/cuda-samples/index.html#cross-samples) for more details on cross platform compilation of cuda samples.
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

## Samples list

### [0. Introduction](./Samples/0_Introduction/README.md)
Basic CUDA samples for beginners that illustrate key concepts with using CUDA and CUDA runtime APIs.

### [1. Utilities](./Samples/1_Utilities/README.md)
Utility samples that demonstrate how to query device capabilities and measure GPU/CPU bandwidth.

### [2. Concepts and Techniques](./Samples/2_Concepts_and_Techniques/README.md)
Samples that demonstrate CUDA related concepts and common problem solving techniques.

### [3. CUDA Features](./Samples/3_CUDA_Features/README.md)
Samples that demonstrate CUDA Features (Cooperative Groups, CUDA Dynamic Parallelism, CUDA Graphs etc).

### [4. CUDA Libraries](./Samples/4_CUDA_Libraries/README.md)
Samples that demonstrate how to use CUDA platform libraries (NPP, NVJPEG, NVGRAPH cuBLAS, cuFFT, cuSPARSE, cuSOLVER and cuRAND).

### [5. Domain Specific](./Samples/5_Domain_Specific/README.md)
Samples that are specific to domain (Graphics, Finance, Image Processing).

### [6. Performance](./Samples/6_Performance/README.md)
Samples that demonstrate performance optimization.

## Dependencies

Some CUDA Samples rely on third-party applications and/or libraries, or features provided by the CUDA Toolkit and Driver, to either build or execute. These dependencies are listed below.

If a sample has a third-party dependency that is available on the system, but is not installed, the sample will waive itself at build time.

Each sample's dependencies are listed in its README's Dependencies section.

### Third-Party Dependencies

These third-party dependencies are required by some CUDA samples. If available, these dependencies are either installed on your system automatically, or are installable via your system's package manager (Linux) or a third-party website.

#### FreeImage

FreeImage is an open source imaging library. FreeImage can usually be installed on Linux using your distribution's package manager system. FreeImage can also be downloaded from the FreeImage website.

To set up FreeImage on a Windows system, extract the FreeImage DLL distribution into the folder `../../../Common/FreeImage/Dist/x64` such that it contains the .h and .lib files. Copy the .dll file to root level `bin/win64/Debug` and `bin/win64/Release` folder.

#### Message Passing Interface

MPI (Message Passing Interface) is an API for communicating data between distributed processes. A MPI compiler can be installed using your Linux distribution's package manager system. It is also available on some online resources, such as [Open MPI](http://www.open-mpi.org/). On Windows, to build and run MPI-CUDA applications one can install [MS-MPI SDK](https://msdn.microsoft.com/en-us/library/bb524831(v=vs.85).aspx).

#### Only 64-Bit

Some samples can only be run on a 64-bit operating system.

#### DirectX

DirectX is a collection of APIs designed to allow development of multimedia applications on Microsoft platforms. For Microsoft platforms, NVIDIA's CUDA Driver supports DirectX. Several CUDA Samples for Windows demonstrates CUDA-DirectX Interoperability, for building such samples one needs to install Microsoft Visual Studio 2012 or higher which provides Microsoft Windows SDK for Windows 8.

#### DirectX12

DirectX 12 is a collection of advanced low-level programming APIs which can reduce driver overhead, designed to allow development of multimedia applications on Microsoft platforms starting with Windows 10 OS onwards. For Microsoft platforms, NVIDIA's CUDA Driver supports DirectX. Few CUDA Samples for Windows demonstrates CUDA-DirectX12 Interoperability, for building such samples one needs to install [Windows 10 SDK or higher](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk), with VS 2015 or VS 2017.

#### OpenGL

OpenGL is a graphics library used for 2D and 3D rendering. On systems which support OpenGL, NVIDIA's OpenGL implementation is provided with the CUDA Driver.

#### OpenGL ES

OpenGL ES is an embedded systems graphics library used for 2D and 3D rendering. On systems which support OpenGL ES, NVIDIA's OpenGL ES implementation is provided with the CUDA Driver.

#### Vulkan

Vulkan is a low-overhead, cross-platform 3D graphics and compute API. Vulkan targets high-performance realtime 3D graphics applications such as video games and interactive media across all platforms. On systems which support Vulkan, NVIDIA's Vulkan implementation is provided with the CUDA Driver. For building and running Vulkan applications one needs to install the [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/).

#### OpenMP

OpenMP is an API for multiprocessing programming. OpenMP can be installed using your Linux distribution's package manager system. It usually comes preinstalled with GCC. It can also be found at the [OpenMP website](http://openmp.org/).

#### Screen

Screen is a windowing system found on the QNX operating system. Screen is usually found as part of the root filesystem.

#### X11

X11 is a windowing system commonly found on *-nix style operating systems. X11 can be installed using your Linux distribution's package manager, and comes preinstalled on Mac OS X systems.

#### EGL

EGL is an interface between Khronos rendering APIs (such as OpenGL, OpenGL ES or OpenVG) and the underlying native platform windowing system.

#### EGLOutput

EGLOutput is a set of EGL extensions which allow EGL to render directly to the display.

#### EGLSync

EGLSync is a set of EGL extensions which provides sync objects that are synchronization primitive, representing events whose completion can be tested or waited upon.

#### NVSCI

NvSci is a set of communication interface libraries out of which CUDA interops with NvSciBuf and NvSciSync. NvSciBuf allows applications to allocate and exchange buffers in memory. NvSciSync allows applications to manage synchronization objects which coordinate when sequences of operations begin and end.

#### NvMedia

NvMedia provides powerful processing of multimedia data for true hardware acceleration across NVIDIA Tegra devices. Applications leverage the NvMedia Application Programming Interface (API) to process the image and video data.

### CUDA Features

These CUDA features are needed by some CUDA samples. They are provided by either the CUDA Toolkit or CUDA Driver. Some features may not be available on your system.

#### CUFFT Callback Routines

CUFFT Callback Routines are user-supplied kernel routines that CUFFT will call when loading or storing data. These callback routines are only available on Linux x86_64 and ppc64le systems.

#### CUDA Dynamic Parallellism

CDP (CUDA Dynamic Parallellism) allows kernels to be launched from threads running on the GPU. CDP is only available on GPUs with SM architecture of 3.5 or above.

#### Multi-block Cooperative Groups

Multi Block Cooperative Groups(MBCG) extends Cooperative Groups and the CUDA programming model to express inter-thread-block synchronization. MBCG is available on GPUs with Pascal and higher architecture.

#### Multi-Device Cooperative Groups

 Multi Device Cooperative Groups extends Cooperative Groups and the CUDA programming model enabling thread blocks executing on multiple GPUs to cooperate and synchronize as they execute. This feature is available on GPUs with Pascal and higher architecture.

#### CUBLAS

CUBLAS (CUDA Basic Linear Algebra Subroutines) is a GPU-accelerated version of the BLAS library.

#### CUDA Interprocess Communication

IPC (Interprocess Communication) allows processes to share device pointers.

#### CUFFT

CUFFT (CUDA Fast Fourier Transform) is a GPU-accelerated FFT library.

#### CURAND

CURAND (CUDA Random Number Generation) is a GPU-accelerated RNG library.

#### CUSPARSE

CUSPARSE (CUDA Sparse Matrix) provides linear algebra subroutines used for sparse matrix calculations.

#### CUSOLVER

CUSOLVER library is a high-level package based on the CUBLAS and CUSPARSE libraries. It combines three separate libraries under a single umbrella, each of which can be used independently or in concert with other toolkit libraries. The intent ofCUSOLVER is to provide useful LAPACK-like features, such as common matrix factorization and triangular solve routines for dense matrices, a sparse least-squares solver and an eigenvalue solver. In addition cuSolver provides a new refactorization library useful for solving sequences of matrices with a shared sparsity pattern.

#### NPP

NPP (NVIDIA Performance Primitives) provides GPU-accelerated image, video, and signal processing functions.

#### NVGRAPH

NVGRAPH is a GPU-accelerated graph analytics library.

#### NVJPEG

NVJPEG library provides high-performance, GPU accelerated JPEG decoding functionality for image formats commonly used in deep learning and hyperscale multimedia applications.

#### NVRTC

NVRTC (CUDA RunTime Compilation) is a runtime compilation library for CUDA C++.

#### Stream Priorities

Stream Priorities allows the creation of streams with specified priorities. Stream Priorities is only available on GPUs with SM architecture of 3.5 or above.

#### Unified Virtual Memory

UVM (Unified Virtual Memory) enables memory that can be accessed by both the CPU and GPU without explicit copying between the two. UVM is only available on Linux and Windows systems.

#### 16-bit Floating Point

FP16 is a 16-bit floating-point format. One bit is used for the sign, five bits for the exponent, and ten bits for the mantissa.

#### C++11 CUDA

NVCC support of [C++11 features](https://en.wikipedia.org/wiki/C++11).

## Contributors Guide

We welcome your input on issues and suggestions for samples. At this time we are not accepting contributions from the public, check back here as we evolve our contribution model.

We use Google C++ Style Guide for all the sources https://google.github.io/styleguide/cppguide.html

## Frequently Asked Questions

Answers to frequently asked questions about CUDA can be found at http://developer.nvidia.com/cuda-faq and in the [CUDA Toolkit Release Notes](http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).

## References

*   [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
*   [Accelerated Computing Blog](https://developer.nvidia.com/blog/?tags=accelerated-computing)

## Attributions

*   Teapot image is obtained from [Wikimedia](https://en.wikipedia.org/wiki/File:Original_Utah_Teapot.jpg) and is licensed under the Creative Commons [Attribution-Share Alike 2.0](https://creativecommons.org/licenses/by-sa/2.0/deed.en) Generic license. The image is modified for samples use cases.