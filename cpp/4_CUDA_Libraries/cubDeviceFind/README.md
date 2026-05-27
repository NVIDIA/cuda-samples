# cubDeviceFind - CUB DeviceFind Search Algorithms

## Description

This sample demonstrates the three device-wide search algorithms: `cub::DeviceFind::FindIf` for predicate search, and `cub::DeviceFind::LowerBound` / `UpperBound` for parallel binary search. Results are verified against `std::find_if`, `std::lower_bound`, and `std::upper_bound` on the host.

## Key Concepts

CCCL 3.3, CUB Device Algorithms, Parallel Search, Binary Search

## Supported SM Architectures

[SM 7.0 ](https://developer.nvidia.com/cuda-gpus) [SM 7.5 ](https://developer.nvidia.com/cuda-gpus) [SM 8.0 ](https://developer.nvidia.com/cuda-gpus) [SM 8.6 ](https://developer.nvidia.com/cuda-gpus) [SM 8.9 ](https://developer.nvidia.com/cuda-gpus) [SM 9.0 ](https://developer.nvidia.com/cuda-gpus) [SM 10.0 ](https://developer.nvidia.com/cuda-gpus) [SM 11.0 ](https://developer.nvidia.com/cuda-gpus) [SM 12.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, aarch64

## CUDA APIs involved

### [CCCL CUB](https://nvidia.github.io/cccl/unstable/cub/index.html)

cub::DeviceFind::FindIf, cub::DeviceFind::LowerBound, cub::DeviceFind::UpperBound

### [CCCL libcu++](https://nvidia.github.io/cccl/unstable/libcudacxx/index.html)

cuda::std::less

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)

cudaDeviceSynchronize, cudaGetDeviceProperties

## Dependencies needed to build/run

[CCCL 3.3+](https://github.com/NVIDIA/cccl). Fetched automatically via CPM at configure time (pinned to `v3.3.3`). Override with `-DCCCL_SOURCE_DIR=/path/to/cccl` to use a local checkout.

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)

[CCCL 3.3 release notes](https://github.com/NVIDIA/cccl/releases), [cub::DeviceFind header](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_find.cuh)
