# clock - Per-Block Kernel Timing with CUB Reduction

## Description

A CUDA sample that demonstrates how to use the `clock()` function to accurately measure kernel execution time on a per-block basis. Each of the 64 blocks records its own start and end SM cycle counter, then performs a block-wide parallel **min-reduction** over 512 float elements using CUB's `BlockReduce`. The host collects all timestamps and computes the average elapsed clock cycles across all blocks.

Because blocks execute in parallel and out of order with no cross-block synchronization, each block independently measures its own execution time — this is the correct way to time GPU work at block granularity.

## What You'll Learn

- Using `clock()` inside a CUDA kernel to capture per-block SM cycle counts
- Performing a block-wide parallel reduction with `cub::BlockReduce` using a custom binary operator
- Loading multiple elements per thread (`ITEMS_PER_THREAD = 2`) for CUB reductions
- Understanding why only thread 0 holds the valid aggregate after a `BlockReduce`
- Querying device properties (`cudaDeviceGetAttribute`) without helper libraries
- Computing average elapsed clocks on the host from per-block timestamps

## Key Concepts

- **SM Clock Counter** — `clock()` reads the streaming multiprocessor's cycle counter; difference between two samples gives elapsed cycles for that block
- **CUB BlockReduce** — warp-shuffle-based block-scope reduction; default constructor allocates shared memory internally via `PrivateStorage()`, no explicit `TempStorage` needed
- **Items per thread** — each thread owns 2 elements; CUB's array overload of `Reduce` combines them before the cross-thread reduction
- **Per-block timing** — since blocks run independently, each block times itself; the host averages results across all blocks

## Key APIs

### CUDA Runtime
- `cudaSetDevice` — select the active GPU for all subsequent CUDA calls
- `cudaDeviceGetAttribute` — query device properties (compute capability, SM count) without `cudaGetDeviceProperties`
- `cudaMalloc` / `cudaFree` — allocate and release device memory
- `cudaMemcpy` — transfer data between host and device
- `clock()` — device-side SM cycle counter (returns `clock_t`)

### CUB
- `cub::BlockReduce<T, BLOCK_THREADS>` — block-scope reduction template
- `BlockReduce::Reduce(T (&input)[ITEMS_PER_THREAD], ReductionOp op)` — reduce multiple items per thread with a custom binary operator

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.5 or higher

### Software
- CMake 3.20 or newer
- A C++17-capable host compiler

## How to Build

See the [top-level README](../../../README.md#building-cuda-samples) for full build instructions, including how to build all samples or a single sample standalone.

## How to Run

```bash
./clock
```

No command-line arguments are required. The sample always runs on device 0.

## Expected Output

```text
CUDA Clock sample
GPU Device 0: with compute capability 8.9 and Number of SMs 142

Average clocks/block = 1239.640625
```

The average clocks value varies by GPU and reflects how many SM cycles each block takes to complete the reduction. Blocks scheduled later on a busy GPU will show higher elapsed times.

## Files

- `clock.cu` — kernel implementation and host driver
- `README.md` — this file
- `CMakeLists.txt` — build configuration

## See Also

- [CUB BlockReduce documentation](https://nvidia.github.io/cccl/unstable/cub/developer/block_scope.html)
- [CUDA C++ Programming Guide — clock()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#clock-and-clock64)
