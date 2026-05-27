# libcuxxMdspan - libcu++ mdspan Interop (DLPack + shared_memory_mdspan)

## Description

This sample demonstrates two mdspan-centric features CCCL: DLPack <-> `cuda::std::mdspan` bridging via `cuda::to_device_mdspan` / `cuda::to_dlpack_tensor` (the tensor-interchange protocol used by PyTorch, JAX, CuPy, and others), and `cuda::shared_memory_mdspan` for multi-dimensional views of shared-memory tiles with address-space-safe accessors. A small matrix is built, wrapped in a DLTensor, converted to a `device_mdspan`, scaled row-wise, and transposed through a `shared_memory_mdspan` tile. The output mdspan is converted back to DLPack and its metadata is printed.

## Key Concepts

CCCL 3.3, libcu++ mdspan, DLPack Interoperability, Shared Memory Views

## Supported SM Architectures

[SM 7.0 ](https://developer.nvidia.com/cuda-gpus) [SM 7.5 ](https://developer.nvidia.com/cuda-gpus) [SM 8.0 ](https://developer.nvidia.com/cuda-gpus) [SM 8.6 ](https://developer.nvidia.com/cuda-gpus) [SM 8.9 ](https://developer.nvidia.com/cuda-gpus) [SM 9.0 ](https://developer.nvidia.com/cuda-gpus) [SM 10.0 ](https://developer.nvidia.com/cuda-gpus) [SM 11.0 ](https://developer.nvidia.com/cuda-gpus) [SM 12.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, aarch64

## CUDA APIs involved

### [CCCL libcu++](https://nvidia.github.io/cccl/libcudacxx/)

cuda::to_device_mdspan, cuda::to_dlpack_tensor, cuda::device_mdspan, cuda::shared_memory_mdspan, cuda::std::mdspan

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)

cudaMalloc, cudaFree, cudaMemcpy, cudaMemset, cudaDeviceSynchronize, cudaGetDeviceProperties

## Dependencies needed to build/run

[CCCL 3.3+](https://github.com/NVIDIA/cccl), [DLPack 1.2+](https://github.com/dmlc/dlpack). Both fetched automatically via CPM at configure time (pinned to `v3.3.3` and `v1.3` respectively). Override with `-DCCCL_SOURCE_DIR=/path/to/cccl` and `-DDLPACK_SOURCE_DIR=/path/to/dlpack` to use local checkouts.

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)

[CCCL 3.3 release notes](https://github.com/NVIDIA/cccl/releases), [cuda::to_device_mdspan header](https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__mdspan/dlpack_to_mdspan.h), [cuda::shared_memory_mdspan docs](https://nvidia.github.io/cccl/libcudacxx/extended_api/mdspan/shared_memory_accessor.html), [DLPack specification](https://dmlc.github.io/dlpack/latest/)
