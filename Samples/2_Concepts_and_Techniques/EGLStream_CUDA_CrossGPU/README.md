# EGLStream_CUDA_CrossGPU - EGLStream_CUDA_CrossGPU

## Description

Demonstrates CUDA and EGL Streams interop, where consumer's EGL Stream is on one GPU and producer's on other and both consumer-producer are different processes.

## Key Concepts

EGLStreams Interop

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
cuEGLStreamConsumerConnect, cuMemFree, cuInit, cuStreamCreate, cuCtxCreate, cuGraphicsResourceGetMappedEglFrame, cuDeviceGetName, cuCtxSynchronize, cuEGLStreamConsumerAcquireFrame, cuDeviceGet, cuDeviceGetAttribute, cuMemAlloc, cuEGLStreamConsumerReleaseFrame, cuEGLStreamProducerDisconnect, cuEGLStreamProducerConnect, cuEGLStreamConsumerDisconnect, cuMemcpyHtoD, cuEGLStreamProducerReturnFrame, cuCtxPushCurrent, cuCtxPopCurrent, cuEGLStreamProducerPresentFrame

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaDeviceCreateConsumer, cudaFree, cudaConsumerReleaseFrame, cudaDeviceSynchronize, cudaGetValueMismatch, cudaProducerDeinit, cudaProducerPresentFrame, cudaMalloc, cudaProducerInit, cudaProducerReturnFrame, cudaProducerPrepareFrame, cudaConsumerAcquireFrame, cudaMemcpy, cudaGetErrorString, cudaDeviceCreateProducer

## Dependencies needed to build/run
[EGL](../../README.md#egl)

## Prerequisites

Download and install the [CUDA Toolkit 11.6](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run

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

