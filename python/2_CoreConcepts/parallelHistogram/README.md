# Sample: Parallel Histogram with Atomics (Python)

## Description

Compute histograms on the GPU using atomic operations to handle concurrent updates from multiple threads. This sample demonstrates the modern **cuda.core API** for kernel compilation and launch, comparing two approaches:

1. **Global Atomics** - All threads atomically update a single global histogram
2. **Privatized Histograms** - Each block uses shared memory, then merges to global

## What You'll Learn

- Compiling CUDA C kernels with `cuda.core.Program`
- Configuring kernel launches with `cuda.core.LaunchConfig`
- Launching kernels with `cuda.core.launch()`
- Using **atomic operations** (`atomicAdd`) for thread-safe updates
- Optimizing with **shared memory privatization**
- GPU timing with `cuda.core` Events

## Key Concepts

### Atomic Operations

When multiple threads update the same histogram bin, a race condition occurs. Atomic operations ensure thread-safe updates:

```cuda
atomicAdd(&histogram[data[i]], 1);  // Thread-safe increment
```

### Global vs Privatized Atomics

| Approach | Pros | Cons |
|----------|------|------|
| Global | Simple | High contention on popular bins |
| Privatized | Significantly faster | Extra shared memory, synchronization |

## Key APIs

### From `cuda.core`:

- `Device` - Device management and context
- `Program` - Compile CUDA C source code
- `ProgramOptions` - Set architecture, optimization flags
- `LaunchConfig` - Configure grid and block dimensions
- `launch()` - Launch compiled kernel
- `Stream` - Async stream management
- `EventOptions` - Configure events for GPU timing
- `stream.record()` - Record events for timing

### From `cupy`:

- `cp.random.randint()` - Generate random data directly on GPU
- `cp.zeros()` - Allocate zeroed GPU arrays

### CUDA Atomic Functions (in kernel):

- `atomicAdd()` - Thread-safe addition

## Requirements

### Hardware:
- NVIDIA GPU with CUDA support

### Software:
- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- See `requirements.txt` for Python packages

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python parallelHistogram.py
```

## Expected Output

```
============================================================
Parallel Histogram with Atomics (cuda.core)
============================================================

Device: <Your GPU>
Compute Capability: ComputeCapability(major=X, minor=Y)

Compiling CUDA kernels with cuda.core.Program...
  Compiled for architecture: sm_XY

Generating 10,000,000 random values on GPU...

Verifying correctness...
  Global atomics:     PASSED
  Privatized atomics: PASSED

Benchmarking (100 iterations)...
  Global atomics:     X.XXX ms
  Privatized atomics: X.XXX ms
  Speedup:            XXx

Test PASSED
```

## Files

- `parallelHistogram.py` - Main sample using cuda.core
- `README.md` - This file
- `requirements.txt` - Dependencies

## See Also

- [cuda.core Documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [CUDA Atomic Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)
- [CUDA Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
