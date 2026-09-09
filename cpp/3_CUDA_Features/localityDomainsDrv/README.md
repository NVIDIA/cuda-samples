# localityDomainsDrv - CUDA Driver API Locality Domains

## Description

This CUDA Driver API sample demonstrates localized compute and memory resources.

Compute is localized by splitting a device's SM resources into locality domain partitions and constructing a green context and stream per locality domain. Memory is localized by creating a localized memory region and mapping it.

## What You'll Learn
How to:
- query a device's locality domain count and SM count per locality domain
- partition SM resources by locality domain ID
- create green contexts and streams from localized SM partitions
- create and map physical allocations in locality domains
- query the locality domain associated with a stream and device pointer

## Key Concepts

- **Locality domain** — a portion of a GPU that contains streaming multiprocessors (SMs) and device memory.
**Green context** — an execution context created from a selected subset of device resources.
- **Virtual memory management** — separate control over virtual addresses, physical allocations, mappings, and access permissions.

## Key APIs

### CUDA Driver

- `cuDeviceGetAttribute` — query the number of locality domains and SMs per domain
- `cuDevSmResourceSplit` — partition the device's SM resources
- `cuMemCreate` — create localized physical memory
- `cuStreamGetDevResource` — inspect stream locality
- `cuPointerGetAttribute` — inspect allocation locality

## Requirements

### Hardware

- NVIDIA GPU that supports locality domains

### Software

- CUDA Toolkit and NVIDIA driver with locality domain support
- Linux or QNX on x86_64 or aarch64
- CMake 3.20 or newer
- A C++17-capable host compiler

## How to Build

See the [top-level README](../../../README.md#building-cuda-samples) for full build instructions, including how to build all samples or a single sample standalone.

## How to Run

```bash
./localityDomainsDrv
```

No command-line arguments are required. The sample always runs on device 0.

## Expected Output

The number of locality domains and SMs depends on the GPU. A run has the following form:

```text
CUDA Driver API locality domains sample
Locality domain count: <domain count>
SMs per locality domain: <SM count>
Device SMs split into <domain count> partitions
 - result[0]: .localityDomainId = 0, .smCount = <SM count>
...
Green context stream 0 is localized to locality domain 0
...
Allocation granularity per chunk: <bytes> bytes
Reserved VA range size: <bytes> bytes
allocation 0x<address> is localized to locality domain 0
...
```

## Files

- `localityDomainsDrv.cpp` — Driver API example
- `README.md` — this file
- `CMakeLists.txt` — build configuration

## See Also

- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
