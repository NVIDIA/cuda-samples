# Sample: Device-Side assert (simpleAssert)

## Description

Use the standard C `assert` macro **inside a CUDA kernel**. The sample launches threads that each assert `gtid < N`; threads whose global index reaches `N` trip the assertion, print a diagnostic message to the host, and cause the kernel to fail. The host detects this through the `cudaErrorAssert` status returned by `cudaDeviceSynchronize` and reports it as the expected outcome.

This sample demonstrates device-side `assert`, a debugging aid for catching invalid conditions in kernel code, and shows how an assertion failure surfaces on the host as an asynchronous CUDA error.

## What You'll Learn

- Calling `assert` directly from device code (`__global__` kernel)
- How a failed device assertion is reported to the host as `cudaErrorAssert`
- Detecting that error after the launch via `cudaDeviceSynchronize`
- Computing a global thread index from `blockIdx`, `blockDim`, and `threadIdx`
- Turning an error code into a human-readable message with `cudaGetErrorString`

## Key Concepts

- **Device-side `assert`** — `assert(condition)` in a kernel; a false condition halts the kernel, prints `file:line: function: block: ... Assertion ... failed`, and flags the launch as failed
- **Asynchronous error reporting** — the assertion failure is not seen at launch time; it is surfaced at the next synchronization point as `cudaErrorAssert`
- **Global thread indexing** — `blockIdx.x * blockDim.x + threadIdx.x`

## Key APIs

### CUDA Runtime
- `cudaSetDevice` — select the active GPU
- `cudaDeviceGetAttribute` — query compute capability (major, minor) and SM count
- `cudaDeviceSynchronize` — block the host until the kernel finishes; flushes assert output and returns `cudaErrorAssert` if an assertion failed
- `cudaGetErrorString` — convert a `cudaError_t` into a readable description

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
./simpleAssert
```

## Expected Output

The kernel launches 2 blocks × 32 threads = 64 threads and asserts `gtid < 60`, so the 4 threads with global indices 60–63 fail the assertion. The assertion failures are **expected** — the sample reports `OK` because it successfully detected `cudaErrorAssert`:

```text
simpleAssert starting...

GPU Device 0: with compute capability X.Y and Number of SMs <smCount>

Launch kernel to generate assertion failures

-- Begin assert output

simpleAssert.cu:50: void simpleAssertKernel(int): block: [1,0,0], thread: [28,0,0] Assertion `gtid < N` failed.
...

-- End assert output

Device assert failed as expected, CUDA error message is: device-side assert triggered

simpleAssert completed, returned OK
```

The order and exact set of failing-thread lines may vary between runs.

## Files

- `simpleAssert.cu` — device-side `assert` kernel + host driver
- `README.md` — this file
- `CMakeLists.txt` — build configuration

## See Also

- [CUDA C++ Programming Guide - Assertion](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#assertion)
