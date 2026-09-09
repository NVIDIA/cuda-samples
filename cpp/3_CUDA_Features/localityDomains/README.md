# localityDomains - CUDA Runtime API Locality Domains

## Description

This CUDA Runtime API sample demonstrates localized compute and memory resources.

Compute is localized by splitting a device's SM resources into locality domain partitions and constructing a green context and stream per locality domain. Memory is localized by allocating from localized memory pools.

## What You'll Learn
How to:
- query a device's locality domain count and SM count per domain
- partition SM resources by locality domain ID
- create green contexts and streams from localized SM partitions
- create localized memory pools and allocate from them asynchronously
- query the locality domain associated with a stream and device pointer

## Key Concepts

- **Locality domain** — a portion of a GPU that contains streaming multiprocessors (SMs) and device memory.
**Green context** — an execution context created from a selected subset of device resources.
- **Localized memory pool** — a stream-ordered allocator whose physical memory is placed in a selected locality domain.

## Key APIs

### CUDA Runtime

- `cudaDeviceGetAttribute` — query the number of locality domains and SMs per domain
- `cudaDevSmResourceSplit` — partition the device's SM resources
- `cudaMemPoolCreate` — create a localized memory pool and allocate from it
- `cudaStreamGetDevResource` — inspect stream locality
- `cudaPointerGetAttributes` — inspect allocation locality

## Requirements

### Hardware

- NVIDIA GPU that supports locality domains and stream-ordered memory pools

### Software

- CUDA Toolkit and NVIDIA driver with locality domain support
- Linux or QNX on x86_64 or aarch64
- CMake 3.20 or newer
- A C++17-capable host compiler

## How to Build

See the [top-level README](../../../README.md#building-cuda-samples) for full build instructions, including how to build all samples or a single sample standalone.

## How to Run

```bash
./localityDomains
```

No command-line arguments are required. The sample always runs on device 0.

## Expected Output

The number of locality domains and SMs depends on the GPU. A run has the following form:

```text
CUDA Runtime API locality domains sample
Locality domain count: <domain count>
SMs per locality domain: <SM count>
Device SMs split into <domain count> partitions
 - result[0]: .localityDomainId = 0, .smCount = <SM count>
...
Green context stream 0 is localized to locality domain 0
...
allocation 0x<address> is localized to locality domain 0
...
```

If device 0 does not support memory pools, the sample reports that it is waiving execution and exits with status EXIT_WAIVED.

## Files

- `localityDomains.cu` — Runtime API example
- `README.md` — this file
- `CMakeLists.txt` — build configuration

## See Also

- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
