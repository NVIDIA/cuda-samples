# Sample: simpleCallback

## Description

A heterogeneous pipeline demo: **CPU pre-process → GPU kernel → CPU post-process**, all
coordinated by a single CUDA stream. A worker thread (`std::thread`) fills the input buffer,
enqueues the GPU work (H2D copy, kernel, D2H copy), and registers a host function with
`cudaLaunchHostFunc` — the modern replacement for the deprecated `cudaStreamAddCallback`.
CUDA invokes that host function automatically once all preceding stream work completes,
so the CPU post-processing (result verification) runs without the main thread ever polling.

## What You'll Learn

- Building a CPU → GPU → CPU pipeline ordered by a single CUDA stream
- Scheduling CPU post-processing on a stream with `cudaLaunchHostFunc`
- Enqueuing GPU work from a `std::thread` worker — all host threads share the device's primary context
- The rule that a host function must never call CUDA APIs
- Why pinned host memory (`cudaMallocHost`) is required for `cudaMemcpyAsync` to be truly asynchronous
- Querying GPU properties (compute capability, SM count) with `cudaDeviceGetAttribute`

## Key Concepts

### Stream-Ordered Pipeline

A stream executes its operations strictly in order. The worker thread pushes the whole
pipeline onto one stream and CUDA handles every dependency:

```text
stream: [H2D copy] → [incrementKernel] → [D2H copy] → [postprocess host func]
```

The main thread only has to `cudaStreamSynchronize` once at the end — when the stream
drains, the results are already verified.

### Host Functions (`cudaLaunchHostFunc`)

A host function is a CPU callback enqueued on a stream like any other operation. CUDA runs
it on an internal thread once all prior work in the stream has finished. Two rules apply:

- It must **not** call any CUDA runtime or driver API (no `cudaFree`, no kernel launches)
- It should be short — the stream cannot proceed until it returns

### Threads Share the Primary Context

With the CUDA runtime API, every host thread in the process shares the device's primary
context. That is why the worker thread can enqueue copies and kernels onto a stream that
`main` created.

### Pinned Host Memory

`cudaMemcpyAsync` only runs *truly asynchronously* when the host buffer is page-locked (pinned),
so the OS cannot move the pages mid-transfer; with ordinary pageable memory the copy falls back to
synchronous behavior. The sample allocates the host buffer with `cudaMallocHost`.

## Key APIs

### CUDA Runtime

- `cudaSetDevice` — select the active GPU for all subsequent CUDA calls
- `cudaDeviceGetAttribute` — query compute capability and SM count
- `cudaMallocHost` / `cudaFreeHost` — allocate and free pinned host memory
- `cudaMalloc` / `cudaFree` — allocate and free device memory
- `cudaStreamCreate` / `cudaStreamDestroy` — create and destroy the stream that orders the pipeline
- `cudaMemcpyAsync` — non-blocking H2D and D2H transfers on the stream
- `cudaLaunchHostFunc` — enqueue a CPU callback on the stream
- `cudaStreamSynchronize` — block the host until the stream (including the host function) is done

### C++ Standard Library

- `std::thread` / `join()` — worker-thread creation

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
./simpleCallback
```

No command-line arguments are required. The sample always runs on device 0.

## Expected Output

```text
=====================================================
  simpleCallback: CPU -> GPU -> CPU pipeline demo
=====================================================
Using GPU 0: compute capability X.Y, <smCount> SMs
[thread]    Stage 1: CPU pre-processing on a worker thread...
[thread]    Filled 100000 elements. First 3 inputs: 42, 43, 44

[thread]    Stage 2: enqueuing H2D copy, kernel, D2H copy on the stream
[thread]    Registering post-processing callback with cudaLaunchHostFunc

[main]      Worker thread joined; GPU work has been enqueued.
[main]      Waiting for the stream (and callback) to finish...

[host func] Stage 3: callback fired automatically - the GPU work is done!
[host func] First 3 results: 43, 44, 45 (each input +1)
[host func] Verified all 100000 results: PASS

[main]      Pipeline complete. Result: SUCCESS
```

**Reading the output:**
- Inputs start at 42 because each element is `workload.id + i` with `id = 42`
- Each result is its input plus one — the kernel's only job
- The `[host func]` lines print from a CUDA internal thread, not from `main`

## Files

- `simpleCallback.cu` — kernel, worker-thread function, host-function callback, and main driver
- `README.md` — this file
- `CMakeLists.txt` — build configuration

## See Also

- [CUDA Programming Guide — Callback Functions from Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#callback-functions-from-streams)
- [CUDA Programming Guide — CUDA Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams)
- [CUDA Runtime API — Execution Control (cudaLaunchHostFunc)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g05841eaa5f90f27124241baafb3e856f)
