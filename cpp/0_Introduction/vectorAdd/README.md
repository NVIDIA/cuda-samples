# Sample: Vector Addition (Unified Memory)

## Description

A first-look CUDA sample: add two float vectors element-wise on the GPU. This version uses **Unified Memory** (`cudaMallocManaged`) so the same pointer is accessible from both host and device, eliminating the need for explicit `cudaMemcpy` calls. The GPU result is verified against a CPU reference computed on the same buffers.

The sample mirrors the corresponding Unified Memory example in the CUDA Programming Guide and uses the `//unified-memory-begin` / `//unified-memory-end` markers so the guide can extract the snippet directly.

## What You'll Learn

- Allocating memory accessible to both host and device with `cudaMallocManaged`
- Launching a 1D CUDA kernel that operates on a Unified Memory buffer with no explicit transfers
- Computing grid dimensions with `cuda::ceil_div` (from `<cuda/cmath>`)
- Synchronizing host execution with device work via `cudaDeviceSynchronize`
- Validating GPU output against a serial CPU reference

## Key Concepts

- **Unified Memory** — one pointer for both CPU and GPU; the CUDA runtime migrates pages on demand
- **1D thread indexing** — `threadIdx.x + blockIdx.x * blockDim.x`
- **Bounds checking** in the kernel so the vector length need not be a multiple of the block size

## Key APIs

### CUDA Runtime
- `cudaMallocManaged` — allocate Unified Memory accessible to both host and device
- `cudaDeviceSynchronize` — block the host until all submitted device work completes
- `cudaFree` — release a Unified Memory allocation

### libcudacxx
- `cuda::ceil_div` — `ceil(a / b)` used to compute the number of blocks; from `<cuda/cmath>`

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.5 or higher

### Software
- CMake 3.20 or newer
- A C++17-capable host compiler

## How to Build

See the [top-level README](../../../README.md#building-cuda-samples) for full build instructions, including how to build all samples or a single sample standalone.

## How to Run

### Basic usage (default vector length = 1024)
```bash
./vectorAdd
```

### Custom vector length
```bash
./vectorAdd 1000000
```

## Expected Output

On success:

```text
Unified Memory: CPU and GPU answers match
```

On failure, the first mismatching index and the two diverging values are printed, followed by an error line:

```text
Index 0 mismatch: 0.123456 != 0.123400
Unified Memory: Error - CPU and GPU answers do not match
```

## Files

- `vectorAdd.cu` — Unified Memory vector addition implementation (kernel + host driver)
- `README.md` — this file
- `CMakeLists.txt` — build configuration

## See Also

- [CUDA C++ Programming Guide — vectorAdd example with Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)

