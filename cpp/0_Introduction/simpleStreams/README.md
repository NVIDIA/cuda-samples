# Sample: simpleStreams

## Description

A simple benchmark that shows how CUDA streams hide memory-transfer latency behind kernel execution.
The GPU computes an element-wise square of a 64 MB float array. It is run twice — once in a single
stream and once split across 4 streams — so you can directly compare the speedup from overlapping
H2D copies, kernel execution, and D2H copies.

## What You'll Learn

- How CUDA streams create independent, in-order queues that enable H2D + kernel + D2H overlap
- How to split a large buffer across N streams so each stream owns its own chunk
- Why pinned (`cudaMallocHost`) host memory is required for `cudaMemcpyAsync` to be truly async
- How to create and use CUDA events to time GPU work with ~0.5 µs precision
- How to query GPU properties (compute capability, SM count) with `cudaDeviceGetAttribute`

## Key Concepts

### CUDA Streams

A stream is an ordered queue of GPU commands. Commands in the **same** stream execute in order;
commands in **different** streams may overlap if the hardware has capacity. This sample uses 4
streams. Each stream owns one quarter of the array and runs its own H2D → kernel → D2H pipeline:

```
Stream 0: [H2D chunk 0][kernel 0][D2H chunk 0]
Stream 1:    [H2D chunk 1][kernel 1][D2H chunk 1]
Stream 2:       [H2D chunk 2][kernel 2][D2H chunk 2]
Stream 3:          [H2D chunk 3][kernel 3][D2H chunk 3]
          |----------------------------------------------> wall clock
```

D2H copy N waits for kernel N (same stream), but overlaps with the H2D copy and kernel of stream N+1.

### The Kernel: `square_kernel`

```cu
out[idx] = in[idx] * in[idx];
```

Each thread computes the square of one float element. Simple and fast — the bottleneck is the
memory transfer, which streams are designed to hide.

### Pinned Host Memory

`cudaMemcpyAsync` requires pinned (page-locked) host memory so the OS cannot swap out the pages
during the transfer. This sample always uses `cudaMallocHost` — the simplest and most portable
pinning strategy.

### Single-stream vs Multi-stream

| Mode | What happens |
|---|---|
| Single stream | H2D → kernel → D2H run sequentially for the full 64 MB array |
| Multi-stream | 4 chunks pipeline concurrently; copy cost approaches 1/4 of the single-stream baseline |

## Key APIs

### CUDA Runtime

- `cudaStreamCreate` / `cudaStreamDestroy` — create and destroy an independent command queue
- `cudaMemcpyAsync` — non-blocking H2D or D2H transfer; requires pinned host memory
- `cudaMemcpy` — blocking transfer used in the single-stream reference benchmark
- `cudaDeviceSynchronize` — wait for all streams to finish before recording the stop event
- `cudaEventCreate` / `cudaEventDestroy` — create and release timing events
- `cudaEventRecord` — insert a timestamp into a stream
- `cudaEventSynchronize` — block the host until the event is recorded
- `cudaEventElapsedTime` — compute milliseconds between two recorded events
- `cudaMallocHost` / `cudaFreeHost` — allocate and free pinned host memory
- `cudaMalloc` / `cudaFree` — standard device memory management
- `cudaDeviceGetAttribute` — query GPU properties such as compute capability and SM count

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.5 or higher

### Software

- CUDA Toolkit (any version supporting the target GPU)
- CMake 3.20 or newer
- A C++17-capable host compiler

## How to Build

See the [top-level README](../../../README.md#building-cuda-samples) for full build instructions, including how to build all samples or a single sample standalone.

## How to Run

```bash
./simpleStreams
```

No arguments needed.

## Expected Output

```text
[ CUDA Sample: Streams ]

GPU Device 0: with compute capability X.Y and Number of SMs <smCount>

Single stream = 8.243 ms
Multi-stream  = 3.167 ms
Speedup       = 2.60x
```

**Reading the numbers:**
- `Single stream` — full 64 MB processed sequentially (H2D + kernel + D2H back-to-back)
- `Multi-stream` — same work split across 4 streams with overlap; typically 2–3x faster
- `Speedup` — ratio of single-stream time to multi-stream time; higher means more overlap achieved

## Files

- `simpleStreams.cu` — kernel, `run_single_stream`, `run_multi_stream`, and main driver
- `README.md` — this file
- `CMakeLists.txt` — build configuration

## See Also

- [CUDA C++ Programming Guide — CUDA Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams)
- [CUDA C++ Programming Guide — Creating and Destroying CUDA Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#creating-and-destroying-cuda-streams)
- [CUDA C++ Best Practices Guide — CUDA Events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events)
