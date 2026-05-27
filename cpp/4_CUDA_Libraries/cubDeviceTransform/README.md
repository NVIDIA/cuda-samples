# cubDeviceTransform - CUB DeviceTransform N-to-M

## Description

This sample demonstrates `cub::DeviceTransform` in its N-input / M-output form. A single device-wide call reads from N input sequences and writes to M output sequences, driven by a user-provided op that returns a `cuda::std::tuple` of M values. Two cases are shown: N=3 inputs producing 1 output, and N=2 inputs producing 2 outputs (sum and difference in one fused pass).

## Key Concepts

CCCL 3.3, CUB Device Algorithms, Fused Elementwise Transforms, Counting Iterators

## Supported SM Architectures

[SM 7.0 ](https://developer.nvidia.com/cuda-gpus) [SM 7.5 ](https://developer.nvidia.com/cuda-gpus) [SM 8.0 ](https://developer.nvidia.com/cuda-gpus) [SM 8.6 ](https://developer.nvidia.com/cuda-gpus) [SM 8.9 ](https://developer.nvidia.com/cuda-gpus) [SM 9.0 ](https://developer.nvidia.com/cuda-gpus) [SM 10.0 ](https://developer.nvidia.com/cuda-gpus) [SM 11.0 ](https://developer.nvidia.com/cuda-gpus) [SM 12.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, aarch64

## CUDA APIs involved

### [CCCL CUB](https://nvidia.github.io/cccl/cub/)

cub::DeviceTransform::Transform

### [CCCL libcu++](https://nvidia.github.io/cccl/libcudacxx/)

cuda::counting_iterator, cuda::std::tuple

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)

cudaDeviceSynchronize, cudaGetDeviceProperties

## Dependencies needed to build/run

[CCCL 3.3+](https://github.com/NVIDIA/cccl). Fetched automatically via CPM at configure time (pinned to `v3.3.3`). Override with `-DCCCL_SOURCE_DIR=/path/to/cccl` to use a local checkout.

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)

[CCCL 3.3 release notes](https://github.com/NVIDIA/cccl/releases), [cub::DeviceTransform header](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_transform.cuh)
