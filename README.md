# CUDA Samples

Samples for CUDA Developers which demonstrates features in CUDA Toolkit. This version supports [CUDA Toolkit 13.0](https://developer.nvidia.com/cuda-downloads).

## Release Notes

This section describes the release notes for the CUDA Samples on GitHub only.

### Change Log

### [Revision History](./CHANGELOG.md)

## Getting Started

### Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
For system requirements and installation instructions of cuda toolkit, please refer to the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/), and the [Windows Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

### Getting the CUDA Samples

Using git clone the repository of CUDA Samples using the command below.
```
git clone https://github.com/NVIDIA/cuda-samples.git
```

Without using git the easiest way to use these samples is to download the zip file containing the current version by clicking the "Download ZIP" button on the repo page. You can then unzip the entire archive and use the samples.

## Building CUDA Samples

### Building CUDA Samples

The CUDA Samples are built using CMake. Follow the instructions below for building on Linux, Windows, and for cross-compilation to Tegra devices.

### Linux

Ensure that CMake (version 3.20 or later) is installed. Install it using your package manager if necessary:

e.g.
```sudo apt install cmake```

Navigate to the root of the cloned repository and create a build directory:
```
mkdir build && cd build
```
Configure the project with CMake:
```
cmake ..
```
Build the samples:
```
make -j$(nproc)
```
Run the samples from their respective directories in the build folder. You can also follow this process from and subdirectory of the samples repo, or from within any individual sample.

### Windows

Language services for CMake are available in Visual Studio 2019 version 16.5 or later, and you can directly import the CUDA samples repository from either the root level or from any
subdirectory or individual sample.

To build from the command line, open the `x64 Native Tools Command Prompt for VS` provided with your Visual Studio installation.

Navigate to the root of the cloned repository and create a build directory:
```
mkdir build && cd build
```
Configure the project with CMake - for example:
```
cmake .. -G "Visual Studio 16 2019" -A x64
```
Open the generated solution file CUDA_Samples.sln in Visual Studio. Build the samples by selecting the desired configuration (e.g., Debug or Release) and pressing F7 (Build Solution).

Run the samples from the output directories specified in Visual Studio.

### Enabling On-GPU Debugging

NVIDIA GPUs support on-GPU debugging through cuda-gdb. Enabling this may significantly affect application performance as certain compiler optimizations are disabled
in this configuration, hence it's not on by default. Enablement of on-device debugging is controlled via the `-G` switch to nvcc.

To enable cuda-gdb for samples builds, define the `ENABLE_CUDA_DEBUG` flag on the CMake command line. For example:

```
cmake -DENABLE_CUDA_DEBUG=True ...
```

### Platform-Specific Samples

Some CUDA samples are specific to certain platforms, and require passing flags into CMake to enable. In particular, we define the following platform-specific flags:

* `BUILD_TEGRA` - for Tegra-specific samples

To build these samples, set the variables either on the command line or through your CMake GUI. For example:

```
cmake -DBUILD_TEGRA=True ..
```

### Cross-Compilation for Tegra Platforms

Install the NVIDIA toolchain and cross-compilation environment for Tegra devices as described in the Tegra Development Guide.

Ensure that CMake (version 3.20 or later) is installed.

Navigate to the root of the cloned repository and create a build directory:
```
mkdir build && cd build
```
Configure the project with CMake, specifying the Tegra toolchain file. And you can use -DTARGET_FS to point to the target file system root path for necessary include and library files:
```
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/toolchain-aarch64-linux.cmake -DTARGET_FS=/path/to/target/system/file/system
```
Build the samples:
```
make -j$(nproc)
```
Transfer the built binaries to the Tegra device and execute them there.


### Cross Building for Automotive Linux Platforms from the DriveOS Docker containers

To build CUDA samples to the target platform from the DriveOS Docker containers, use the following instructions.

Mount the target Root Filesystem (RFS) in the container so that the CUDA cmake process has the correct paths to CUDA and other system libraries required to build the samples.

Create a temporary directory, `<temp>` is any temporary directory of your choosing, for example, you can use `/drive/temp`:

```
$ mkdir /drive/<temp>
```

Mount the filesystem by running the following command:

```
$ mount /drive/drive-linux/filesystem/targetfs-images/dev_nsr_desktop_ubuntu-24.04_thor_rfs.img /drive/temp
```

Configure the project by running the following cmake command:

```
$ mkdir build && cd build
$ cmake .. -DBUILD_TEGRA=True \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/toolchain-aarch64-linux.cmake \
  -DTARGET_FS=/drive/temp \
  -DCMAKE_LIBRARY_PATH=/drive/temp/usr/local/cuda-13.0/thor/lib64/ \
  -DCMAKE_INCLUDE_PATH=/drive/temp/usr/local/cuda-13.0/thor/include/
```

Please note that the following libraries are not pre-installed in the DriveOS dev-nsr target filesystem:
* libdrm-dev
* Vulkan

This causes the cmake command to throw errors related to the missing files, and as a result, the related samples will not build in later steps. This issue will be addressed in a future DriveOS release.

To build the samples with ignore the error mentioned above, you can use `--ignore-errors`/`--keep-going` or comment out the comment out the corresponding `add_subdirectory` command in the CMakeLists.txt in the parent folder for the samples requiring Vulkan and libdrm_dev:

```
$ make -j$(nproc) --ignore-errors # or --keep-going
```

```
# In Samples/5_Domain_Specific/CMakeList.txt
# add_subdirectory(simpleGL)
# add_subdirectory(simpleVulkan)
# add_subdirectory(simpleVulkanMMAP)

# In Samples/8_Platform_Specific/Tegra/CMakeList.txt
# add_subdirectory(simpleGLES_EGLOutput)
```

### QNX

Cross-compilation for QNX with CMake is supported in the CUDA 13.0 samples release and newer. An example build for
the Tegra Thor QNX platform might look like this:

```
$ mkdir build
$ cd build

QNX_HOST=/path/to/qnx/host \
QNX_TARGET=/path/to/qnx/target \
cmake .. \
-DBUILD_TEGRA=True \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-safe-13.0/bin/nvcc \
-DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/toolchain-aarch64-qnx.cmake \
-DCMAKE_LIBRARY_PATH=/usr/local/cuda-safe-13.0/thor/targets/aarch64-qnx/lib/stubs/ \
-DCMAKE_INCLUDE_PATH=/usr/local/cuda-safe-13.0/thor/targets/aarch64-qnx/include/
```

### Forward Compatibility

To build samples with new CUDA Toolkit(CUDA 13.0 or later) and UMD(Version 580 or later) and old KMD(Version 550 or earlier)，you need to set the `CMAKE_PREFIX_PATH` for using new driver library, the command might like this:

```
cmake -DCMAKE_PREFIX_PATH=/usr/local/cuda/lib64/stubs/ ..
```

## Running All Samples as Tests

It's important to note that the CUDA samples are _not_ intended as a validation suite for CUDA. They do not cover corner cases, they do not completely cover the
runtime and driver APIs, are not intended for performance benchmarking, etc. That said, it can sometimes be useful to run all of the samples as a quick sanity check and
we provide a script to do so, `run_tests.py`.

This Python3 script finds all executables in a subdirectory you choose, matching application names with command line arguments specified in `test_args.json`. It accepts
the following command line arguments:

| Switch     | Purpose                                                                                                        | Example                 |
| ---------- | -------------------------------------------------------------------------------------------------------------- | ----------------------- |
| --dir      | Specify the root directory to search for executables (recursively)                                             | --dir ./build/Samples   |
| --config   | JSON configuration file for executable arguments                                                               | --config test_args.json |
| --output   | Output directory for test results (stdout saved to .txt files - directory will be created if it doesn't exist) | --output ./test         |
| --args     | Global arguments to pass to all executables (not currently used)                                               | --args arg_1 arg_2 ...  |
| --parallel | Number of applications to execute in parallel.                                                                 | --parallel 8            |


Application configurations are loaded from `test_args.json` and matched against executable names (discarding the `.exe` extension on Windows).

The script returns 0 on success, or the first non-zero error code encountered during testing on failure. It will also print a condensed list of samples that failed, if any.

There are three primary modes of configuration:

**Skip**

An executable configured with "skip" will not be executed. These generally rely on having attached graphical displays and are not suited to this kind of automation.

Configuration example:
```json
"fluidsGL": {
    "skip": true
}
```

You will see:
```
Skipping fluidsGL (marked as skip in config)
```

**Single Run**

For executables to run one time only with arguments, specify each argument as a list entry. Each entry in the JSON file will be appended to the command line, separated
by a space.

All applications execute from their current directory, so all paths are relative to the application's location.

Note that if an application needs no arguments, this entry is optional. An executable found without a matching entry in the JSON will just run as `./application` from its
current directory.

Configuration example:
```json
"ptxgen": {
    "args": [
        "test.ll",
        "-arch=compute_75"
    ]
}
```

You will see:
```
Running ptxgen
    Command: ./ptxgen test.ll -arch=compute_75
    Test completed with return code 0
```

**Multiple Runs**

For executables to run multiple times with different command line arguments, specify any number of sets of args within a "runs" list.

As with single runs, all applications execute from their current directory, so all paths are relative to the application's location.

Configuration example:
```json
"recursiveGaussian": {
    "runs": [
        {
            "args": [
                "-sigma=10",
                "-file=data/ref_10.ppm"
            ]
        },
        {
            "args": [
                "-sigma=14",
                "-file=data/ref_14.ppm"
            ]
        },
        {
            "args": [
                "-sigma=18",
                "-file=data/ref_18.ppm"
            ]
        },
        {
            "args": [
                "-sigma=22",
                "-file=data/ref_22.ppm"
            ]
        }
    ]
}
```

You will see:
```
Running recursiveGaussian (run 1/4)
    Command: ./recursiveGaussian -sigma=10 -file=data/ref_10.ppm
    Test completed with return code 0
Running recursiveGaussian (run 2/4)
    Command: ./recursiveGaussian -sigma=14 -file=data/ref_14.ppm
    Test completed with return code 0
Running recursiveGaussian (run 3/4)
    Command: ./recursiveGaussian -sigma=18 -file=data/ref_18.ppm
    Test completed with return code 0
Running recursiveGaussian (run 4/4)
    Command: ./recursiveGaussian -sigma=22 -file=data/ref_22.ppm
    Test completed with return code 0
```

### Example Usage

Here is an example set of commands to build and test all of the samples.

First, build:
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

Now, return to the samples root directory and run the test script:
```bash
cd ..
python3 run_tests.py --output ./test --dir ./build/Samples --config test_args.json
```

If all applications run successfully, you will see something similar to this (the specific number of samples will depend on your build type
and system configuration):

```
Test Summary:
Ran 199 test runs for 180 executables.
All test runs passed!
```

If some samples fail, you will see something like this:

```
Test Summary:
Ran 199 test runs for 180 executables.
Failed runs (2):
  bicubicTexture (run 1/5): Failed (code 1)
  Mandelbrot (run 1/2): Failed (code 1)
```

You can inspect the stdout logs in the output directory (generally `APM_<application_name>.txt` or `APM_<application_name>.run<n>.txt`) to help
determine what may have gone wrong from the output logs. Please file issues against the samples repository if you believe a sample is failing
incorrectly on your system.

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

### [7. libNVVM](./Samples/7_libNVVM/README.md)
Samples that demonstrate the use of libNVVVM and NVVM IR.

## Dependencies

Some CUDA Samples rely on third-party applications and/or libraries, or features provided by the CUDA Toolkit and Driver, to either build or execute. These dependencies are listed below.

If a sample has a third-party dependency that is available on the system, but is not installed, the sample will waive itself at build time.

Each sample's dependencies are listed in its README's Dependencies section.

### Third-Party Dependencies

These third-party dependencies are required by some CUDA samples. If available, these dependencies are either installed on your system automatically, or are installable via your system's package manager (Linux) or a third-party website.

#### FreeImage

FreeImage is an open source imaging library. FreeImage can usually be installed on Linux using your distribution's package manager system. FreeImage can also be downloaded from the FreeImage website.

To set up FreeImage on a Windows system, extract the FreeImage DLL distribution into the folder `./Common/FreeImage/Dist/x64` such that it contains the .h and .lib files. Copy the .dll file to the Release/ Debug/ execution folder or pass the FreeImage folder when cmake configuring with the `-DFreeImage_INCLUDE_DIR` and `-DFreeImage_LIBRARY` options.

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

#### GLFW
GLFW is a lightweight, open-source library designed for managing OpenGL, OpenGL ES, and Vulkan contexts. It simplifies the process of creating and managing windows, handling user input (keyboard, mouse, and joystick), and working with multiple monitors in a cross-platform manner.

To set up GLFW on a Windows system, Download the pre-built binaries from [GLFW website](https://www.glfw.org/download.html) and extract the zip file into the folder, pass the GLFW include header folder as `-DGLFW_INCLUDE_DIR` and lib folder as `-DGLFW_LIB_DIR` for cmake configuring.

#### OpenMP

OpenMP is an API for multiprocessing programming. OpenMP can be installed using your Linux distribution's package manager system. It usually comes preinstalled with GCC. It can also be found at the [OpenMP website](http://openmp.org/). For compilers such as clang, `libomp.so` and other components for LLVM must be installed separated. You will also need to set additional flags in your CMake configuration files, such as: `-DOpenMP_CXX_FLAGS="-fopenmp=libomp" -DOpenMP_CXX_LIB_NAMES="omp" -DOpenMP_omp_LIBRARY="/path/to/libomp.so"`.

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

#### CMake

The libNVVM samples are built using [CMake](https://cmake.org/) 3.10 or later.

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
