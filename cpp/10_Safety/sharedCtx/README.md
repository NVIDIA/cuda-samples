# sharedCtx - Shared Device Context (CUDA Safe Toolkit)

## Description

This sample shows how two CUDA processes, each with its own CUDA primary context, can share one set of GPU scheduling resources. Because both primary contexts are scheduled through the same GPU context, the GPU does not swap scheduling resources on and off when it switches between work from the two processes. Kernels launched from the two processes can also execute concurrently. The sample takes no arguments and launches the second process itself.

1. Process A calls `cudaSafeExQueryDeviceKey()` before any CUDA context is created in it, and sends the resulting device key to process B.
2. Process B creates its primary context, then calls `cudaSafeExSharedDeviceCtxKeyExport()` with that device key.
3. After process B has exported, process A calls `cudaSafeExSharedDeviceCtxKeyImport()`, then initializes its primary context. This primary context is created on the same GPU scheduling resources as process B's primary context.
4. Each process launches a small kernel on its own primary context.

`cudaSafeExQueryDeviceKey()` and `cudaSafeExSharedDeviceCtxKeyImport()` must be called before a CUDA context is created in the calling process, including the primary context created implicitly by the first CUDA Runtime API call that triggers context creation. `cudaSafeExSharedDeviceCtxKeyExport()` must be called after the calling process has created its primary context.

## Why this is safety-specific

These three calls are safety extensions of the CUDA runtime. They are declared in `cuda_runtime_api_safe_ex.h`, which ships with the CUDA Safe toolkit, and they are allowed only while CUDA is in the Initialization state.

In standard CUDA, this functionality is exposed through the CUDA MPS feature.

## Key Concepts

CUDA Safe Toolkit, Shared Device Context, Multi-Process

## Supported SM Architectures

[SM 11.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

QNX Safety (CUDA Safe toolkit cross-build)


## Prerequisites

Download and install the CUDA Safe toolkit for QNX Safety. It is skipped when `cuda_runtime_api_safe_ex.h` is not in the toolkit include path. It does not need `TARGET_FS`.

Build it from `cpp/10_Safety/sharedCtx` with the QNX Safety toolchain. Set `CMAKE_CUDA_ARCHITECTURES` to `110` for Thor. See the [safety samples README](../README.md).
