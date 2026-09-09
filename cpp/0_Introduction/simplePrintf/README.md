# Sample: Device-Side printf (simplePrintf)

## Description

A first-look CUDA sample: call C's `printf` **from inside a CUDA kernel**. Every thread in a 2D grid of 3D blocks prints its own block and thread identifiers along with a shared value passed from the host. The CUDA runtime buffers the device-side output and flushes it to the host's standard output when the kernel completes.

This sample is the canonical demonstration of device-side `printf`, the simplest debugging and tracing tool available inside kernel code. It also shows how multidimensional `blockIdx`/`threadIdx` coordinates are flattened into linear indices.

## What You'll Learn

- Calling `printf` directly from device code (`__global__` kernel)
- Launching a kernel with a **2D grid** (`dim3 dimGrid(2, 2)`) of **3D blocks** (`dim3 dimBlock(2, 2, 2)`)
- Flattening multidimensional block and thread indices into a single linear index
- Flushing device-side `printf` output to the host with `cudaDeviceSynchronize`
- Querying the active device with `cudaGetDevice` / `cudaGetDeviceProperties`

## Key Concepts

- **Device-side `printf`** — formatted output from within a kernel; output is buffered per launch and flushed at a synchronization point
- **Multidimensional launch geometry** — `gridDim`, `blockIdx`, `blockDim`, and `threadIdx` are `dim3` values with `.x`, `.y`, `.z` components
- **Index linearization** — converting `(x, y, z)` coordinates into a flat index:
  - block: `blockIdx.y * gridDim.x + blockIdx.x`
  - thread: `threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x`

## Key APIs

### CUDA Runtime
- `cudaSetDevice` — select the active GPU
- `cudaDeviceGetAttribute` — query compute capability (major, minor) and SM count
- `cudaDeviceSynchronize` — block the host until the kernel finishes, which also flushes the device `printf` buffer

### Device
- `printf` — standard C formatted output, callable from `__global__`/`__device__` code (requires Compute Capability 2.0 or higher)

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 2.0 or higher (device-side `printf` is unavailable on earlier architectures)

### Software
- CMake 3.20 or newer
- A C++17-capable host compiler

## How to Build

See the [top-level README](../../../README.md#building-cuda-samples) for full build instructions, including how to build all samples or a single sample standalone.

## How to Run

```bash
./simplePrintf
```

## Expected Output

The launch uses a 2×2 grid of 2×2×2 blocks = 4 blocks × 8 threads = 32 lines. Each thread prints its linear block index, its linear thread index, and the value `10`. Because blocks and threads run concurrently, **the order of the lines will vary between runs**:

```text
GPU Device 0: with compute capability X.Y and Number of SMs <smCount>
printf() is called. Output:

[0, 0]:		Value is:10
[0, 1]:		Value is:10
[0, 2]:		Value is:10
...
[3, 7]:		Value is:10
```

## Files

- `simplePrintf.cu` — device-side `printf` kernel + host driver
- `README.md` — this file
- `CMakeLists.txt` — build configuration

## See Also

- [CUDA C++ Programming Guide — printf()](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#printf)
