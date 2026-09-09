# Sample: simpleMultiGPU

## Description

A multi-GPU reduction demo: the input vector is split across every GPU in the system, each
GPU sums its slice concurrently, and the host combines the per-GPU results. Each GPU gets its
own CUDA stream, so the H2D copy → reduction kernel → D2H copy pipelines run in parallel across
devices. The device-side reduction uses `cub::BlockReduce`; the host finishes the sum across
blocks and across GPUs and checks it against a CPU reference. The whole GPU phase is timed with
CUDA events.

## What You'll Learn

- Enumerating and driving multiple GPUs with `cudaGetDeviceCount` and `cudaSetDevice`
- Partitioning a workload across GPUs and running them concurrently, one CUDA stream per GPU
- Overlapping H2D copy, kernel, and D2H copy with `cudaMemcpyAsync` on independent streams
- Reducing within a block using `cub::BlockReduce`, then finishing the reduction on the host
- Timing GPU work with CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`)
- Why pinned host memory (`cudaMallocHost`) is required for `cudaMemcpyAsync` to be asynchronous

## Key Concepts

### One Stream per GPU

The work is held in a `std::vector<TGPUplan>` with one entry per device, each carrying its own stream,
buffers, and data slice. Selecting a device with `cudaSetDevice` and issuing async work on that
device's stream lets all GPUs run at the same time:

```text
GPU 0 stream: [H2D slice 0] → [reduceKernel] → [D2H partials 0]
GPU 1 stream: [H2D slice 1] → [reduceKernel] → [D2H partials 1]
...
```

The host launches every GPU's pipeline before synchronizing any of them, so the copies and kernels
overlap across devices.

### Multi-Level Reduction

The reduction happens in three stages. On each GPU, `reduceKernel` launches `BLOCK_N` blocks of `THREAD_N`
threads: every thread accumulates a grid-strided partial sum, then `cub::BlockReduce` combines the
per-thread partials within its block and thread 0 writes one value per block. The host then adds the
`BLOCK_N` per-block partials into that GPU's sum, and finally adds the per-GPU sums together.

### Pinned Host Memory

`cudaMemcpyAsync` only runs asynchronously when the host buffer is page-locked (pinned), so the OS
cannot move the pages mid-transfer; with ordinary pageable memory the copy falls back to synchronous
behavior. Both the input slice and the partial-sum buffer are allocated with `cudaMallocHost`.

### Timing with CUDA Events

CUDA events are recorded on device 0's stream to bracket the GPU phase. `cudaEventElapsedTime` returns
the milliseconds between the recorded events.

## Key APIs

### CUDA Runtime

- `cudaGetDeviceCount` — count the CUDA-capable GPUs in the system
- `cudaSetDevice` — select the active GPU for subsequent CUDA calls
- `cudaStreamCreate` / `cudaStreamDestroy` — per-GPU stream that orders each pipeline
- `cudaMalloc` / `cudaFree` — allocate and free device memory
- `cudaMallocHost` / `cudaFreeHost` — allocate and free pinned host memory
- `cudaMemcpyAsync` — non-blocking H2D and D2H transfers on a stream
- `cudaStreamSynchronize` — block the host until a GPU's stream is done
- `cudaEventCreate` / `cudaEventRecord` / `cudaEventSynchronize` / `cudaEventElapsedTime` / `cudaEventDestroy` — event-based timing

### CCCL / CUB

- `cub::BlockReduce` — block-wide reduction inside the kernel

## Requirements

### Hardware

- Two or more NVIDIA GPUs with Compute Capability 7.5 or higher

### Software

- CMake 3.20 or newer
- A C++17-capable host compiler

## How to Build

See the [top-level README](../../../README.md#building-cuda-samples) for full build instructions, including how to build all samples or a single sample standalone.

## How to Run

```bash
./simpleMultiGPU
```

No command-line arguments are required. The sample uses every GPU it detects, and exits without
running if it finds fewer than two.

## Expected Output

```text
Starting simpleMultiGPU
CUDA-capable device count: 2
Generating input data...

Computing with 2 GPUs...
  GPU Processing time: 8.138752 (ms)

Computing with Host CPU...

Comparing GPU and Host CPU results...
  GPU sum: 16777294.000000
  CPU sum: 16777294.395033
  Relative difference: 2.354566E-08
```

**Reading the output:**
- The device count and processing time depend on your system
- The sums are the same every run because the input is filled from an unseeded `rand()`
- The sample passes when the GPU and CPU results agree to within a relative difference of `1e-5`

## Files

- `simpleMultiGPU.cu` — reduction kernel and the multi-GPU driver
- `README.md` — this file
- `CMakeLists.txt` — build configuration

## See Also

- [CUDA Programming Guide — CUDA Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams)
- [CUDA Programming Guide — CUDA Events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events)
- [CUDA Core Compute Libraries  —  What is CUB?](https://nvidia.github.io/cccl/unstable/cub/index.html#what-is-cub)
