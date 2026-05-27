# libcuxxRandom - libcu++ Random Distributions

## Description

This sample demonstrates the random-number facilities added to libcu++ in CCCL. `<cuda/std/random>` now offers host- and device-compatible implementations of the standard C++ distributions (uniform, normal, Poisson, Bernoulli, and more) and backports the C++26 `cuda::std::philox4x32` / `philox4x64` engines. `<cuda/random>` adds `cuda::pcg64` as an NVIDIA extension (the same generator NumPy uses by default). A kernel draws samples on each thread and the host computes empirical statistics, comparing them to the theoretical mean / variance / probability.

## Key Concepts

CCCL 3.3, libcu++ Random, PCG, Philox, Device-Side PRNG

## Supported SM Architectures

[SM 7.0 ](https://developer.nvidia.com/cuda-gpus) [SM 7.5 ](https://developer.nvidia.com/cuda-gpus) [SM 8.0 ](https://developer.nvidia.com/cuda-gpus) [SM 8.6 ](https://developer.nvidia.com/cuda-gpus) [SM 8.9 ](https://developer.nvidia.com/cuda-gpus) [SM 9.0 ](https://developer.nvidia.com/cuda-gpus) [SM 10.0 ](https://developer.nvidia.com/cuda-gpus) [SM 11.0 ](https://developer.nvidia.com/cuda-gpus) [SM 12.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, aarch64

## CUDA APIs involved

### [CCCL libcu++](https://nvidia.github.io/cccl/libcudacxx/)

cuda::pcg64, cuda::std::philox4x32, cuda::std::uniform_real_distribution, cuda::std::normal_distribution, cuda::std::poisson_distribution, cuda::std::bernoulli_distribution

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)

cudaMalloc, cudaFree, cudaMemcpy, cudaDeviceSynchronize, cudaGetDeviceProperties

## Dependencies needed to build/run

[CCCL 3.3+](https://github.com/NVIDIA/cccl). Fetched automatically via CPM at configure time (pinned to `v3.3.3`). Override with `-DCCCL_SOURCE_DIR=/path/to/cccl` to use a local checkout.

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)

[CCCL 3.3 release notes](https://github.com/NVIDIA/cccl/releases), [cuda::pcg64 header](https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__random/pcg_engine.h), [NumPy PCG64](https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html)
