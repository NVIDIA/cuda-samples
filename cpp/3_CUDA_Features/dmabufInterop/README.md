# Sample: CUDA dma-buf Interoperability

## Description

Linux **dma-buf** is the kernel's standard mechanism for sharing a buffer between subsystems by representing it as a file descriptor. This sample shows CUDA acting as both a dma-buf **producer** (taking one of its own allocations and obtaining a dma-buf fd for it with `cuMemGetHandleForAddressRange`) and a dma-buf **consumer** (taking an existing dma-buf fd and turning it into a CUDA-mappable buffer with `cuImportExternalMemory` + `cuExternalMemoryGetMappedBuffer`). A single binary runs three demos in sequence:

1. **Same-process round-trip** — one CUDA context exports a host allocation and imports it back as external memory.
2. **Cross-process IPC** — a `fork()`-ed child exports an allocation and passes the fd to the parent over a Unix domain socket using `SCM_RIGHTS`.
3. **Cross-GPU sharing across processes** — the same `fork()` + `SCM_RIGHTS` path as demo 2, but the producer child runs on one GPU and the consumer parent imports the fd into a different GPU's context. The sample auto-detects any pair of GPUs whose driver reports support for the dma-buf capability attributes. This demo self-skips when no qualifying pair is present.

Producer-side allocations always use `cuMemAllocHost`; using a single allocator keeps the producer code identical across all three demos.

## What You'll Learn

- Exporting a CUDA host allocation as a Linux dma-buf file descriptor
- Importing a dma-buf fd into CUDA as external memory and mapping it to a device pointer
- Passing a dma-buf fd between processes via a Unix-domain socket using `SCM_RIGHTS`
- Sharing a single buffer between two GPUs in two different processes, with the producer-completes-before-consumer-reads ordering enforced by the existing fork + socket handshake
- Detecting dma-buf platform support at runtime with CUDA device attributes
- Coordinating fork()ed children so output and CUDA work appear in source order rather than the order processes happen to be scheduled

## Key Concepts

- **dma-buf** — kernel-side shareable buffer represented as a file descriptor; usable across CUDA, V4L2, DRM/KMS, Vulkan, and any other subsystem that speaks dma-buf
- **Page-aligned export** — `cuMemGetHandleForAddressRange` requires a page-aligned base, so the producer over-allocates by one page and rounds up
- **fd as IPC primitive** — once CUDA has produced a dma-buf fd, the fd is just a POSIX file descriptor and can be sent across `fork()` boundaries via `SCM_RIGHTS`
- **Capability-gated execution** — `CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED` (producer side) and `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` (consumer side) determine which paths run on the current hardware
- **Parent-arbitrated synchronisation** — both fork()ed children block on a one-byte "go" from the parent before doing any CUDA work, so output across the three demos appears in source order even when stdout is fully buffered (e.g., over ssh)

## Key APIs

### CUDA Driver API
- `cuMemAllocHost` — allocate page-locked host memory that is also a valid CUDA device pointer
- `cuMemGetHandleForAddressRange` (with `CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD`) — export a CUDA allocation as a Linux dma-buf fd
- `cuImportExternalMemory` (with `CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD`) — bring a dma-buf fd into CUDA as external memory
- `cuExternalMemoryGetMappedBuffer` — obtain a `CUdeviceptr` from an imported external memory object
- `cuDeviceGetAttribute` — runtime capability query for `CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED` and `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED`

### CUDA Runtime API
- `cudaSetDevice` — pin the runtime to the producer GPU (child process) or the consumer GPU (parent process); both demos that use multiple GPUs do this on each side after the fork
- `cudaGetDeviceCount` — used inside the cross-GPU pair detection to enumerate the available GPUs
- `cudaDeviceSynchronize` — used on the producer side to make sure the kernel that wrote the buffer has retired before the fd is sent to the consumer

### Linux / POSIX
- `socketpair` + `fork` + `sendmsg`/`recvmsg` with `SCM_RIGHTS` — pass a file descriptor between processes

## Requirements

### Hardware
- An NVIDIA GPU whose `CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED` is `1`. The sample queries this at startup and skips entirely on unsupported devices.
- The cross-GPU section additionally needs at least two GPUs in the system, with one reporting `CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED = 1` (producer) and the other reporting `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 1` (consumer). On systems without a qualifying pair the cross-GPU section prints a skip message and the sample continues.

### Software
- CUDA Toolkit 13.4 or newer
- CMake 3.20 or newer
- Linux. dma-buf is a Linux kernel facility; this sample does not build on Windows or QNX.

## How to Build

See the [top-level README](../../../README.md#building-cuda-samples) for full build instructions, including how to build all samples or a single sample standalone.

## How to Run

The sample takes no arguments:

```bash
./dmabufInterop
```

All three sections run in sequence. The cross-GPU section self-skips when the running system has no qualifying GPU pair, so the same binary is appropriate for a single-GPU workstation, a multi-GPU server, or a heterogeneous-GPU platform.

## Expected Output

On a system with a single discrete GPU (the cross-GPU section self-skips):

```text
Device 0: <GPU name> (Compute Capability X.Y)

Same-process round-trip
  producer: allocated 16384 bytes, exported as dma-buf fd 45
  producer: wrote pattern
  consumer: imported dma-buf as device pointer
  consumer: verified (0 mismatches)

Cross-process IPC (fork + SCM_RIGHTS)
  child:  allocated 16384 bytes, exported as dma-buf fd 45
  child:  wrote pattern
  child:  sent fd to parent
  parent: received dma-buf fd 45 from child
  parent: imported dma-buf as device pointer
  parent: verified (0 mismatches)

Cross-GPU sharing across processes
  no GPU pair satisfies the dma-buf capability attributes. Skipping.

Done
```

On a system with two or more dma-buf-capable GPUs, the cross-GPU section runs to completion:

```text
Cross-GPU sharing across processes
  child:  on device 1, allocated 16384 bytes, exported as dma-buf fd 53
  child:  wrote pattern
  child:  sent fd to parent
  parent: on device 0, received dma-buf fd 4 from child
  parent: imported dma-buf as device pointer
  parent: verified (0 mismatches)
```

If a verification kernel reports any mismatches, the corresponding line ends with `Error (N mismatches)` and the sample exits with a non-zero status.

## Files

- `dmabufInterop.cu` — kernels, CUDA export/import helpers, and the three demo functions. Reading this file end-to-end gives the full CUDA dma-buf flow without needing to open anything else.
- `dmabufInterop_helpers.h` — POSIX-only helpers used by the cross-process demos: `sendFd` / `recvFd` for `SCM_RIGHTS`-based file-descriptor passing, and `reapCrossProcessChild` (a `waitpid` wrapper). Split out so the main `.cu` stays focused on CUDA; open this file only if you want to see the SCM_RIGHTS mechanics.
- `README.md` — this file
- `CMakeLists.txt` — build configuration (Linux-only guard, CUDA driver + runtime link)

## See Also

- [Linux kernel dma-buf documentation](https://docs.kernel.org/driver-api/dma-buf.html)
- [CUDA External Resource Interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html)
- [`cuMemGetHandleForAddressRange`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
