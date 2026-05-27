# cubDeviceSegmentedScan - CUB DeviceSegmentedScan

## Description

This sample demonstrates `cub::DeviceSegmentedScan`. A segmented scan computes an independent scan over each of many contiguous segments in a single device-wide call. Two operations are shown: `ExclusiveSegmentedSum` across three independent segments, and `InclusiveSegmentedScan` with a custom binary operator (running maximum via `cuda::maximum<>`).

## Key Concepts

CUB Device Algorithms, Segmented Scan, Prefix Sum

## Supported SM Architectures

[SM 7.0 ](https://developer.nvidia.com/cuda-gpus) [SM 7.5 ](https://developer.nvidia.com/cuda-gpus) [SM 8.0 ](https://developer.nvidia.com/cuda-gpus) [SM 8.6 ](https://developer.nvidia.com/cuda-gpus) [SM 8.9 ](https://developer.nvidia.com/cuda-gpus) [SM 9.0 ](https://developer.nvidia.com/cuda-gpus) [SM 10.0 ](https://developer.nvidia.com/cuda-gpus) [SM 11.0 ](https://developer.nvidia.com/cuda-gpus) [SM 12.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, aarch64

## CUDA APIs involved

### [CCCL CUB](https://nvidia.github.io/cccl/cub/)

cub::DeviceSegmentedScan::ExclusiveSegmentedSum, cub::DeviceSegmentedScan::InclusiveSegmentedScan

### [CCCL libcu++](https://nvidia.github.io/cccl/libcudacxx/)

cuda::maximum

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)

cudaDeviceSynchronize, cudaGetDeviceProperties

## Dependencies needed to build/run

[CCCL 3.3+](https://github.com/NVIDIA/cccl). Fetched automatically via CPM at configure time (pinned to `v3.3.3`). Override with `-DCCCL_SOURCE_DIR=/path/to/cccl` to use a local checkout.

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)

[CCCL 3.3 release notes](https://github.com/NVIDIA/cccl/releases), [cub::DeviceSegmentedScan header](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_segmented_scan.cuh)
