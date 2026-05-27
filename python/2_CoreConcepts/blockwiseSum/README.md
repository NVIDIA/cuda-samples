# Sample: Block-wise Array Sum (Python)

## Description

Demonstrates fundamental CUDA thread cooperation: thread/block indexing, strided loops, and block-wise reduction using shared memory. This sample shows three progressively complex kernel patterns using the **cuda.core API**:

1. **Simple indexing** - One thread per element
2. **Strided loop** - Each thread processes multiple elements
3. **Block partial sum** - Shared memory reduction within each block

## What You'll Learn

- How to calculate global thread ID from block and thread indices
- Strided loop pattern for processing arrays larger than grid size
- Block-level cooperation using shared memory and `__syncthreads()`

## Key Concepts

### Thread and Block Indexing

```
Global Thread ID = blockIdx.x * blockDim.x + threadIdx.x
Stride = blockDim.x * gridDim.x
```

### Strided Loop Pattern

Each thread processes multiple elements, enabling fixed grid size for arbitrary array lengths:

```c
for (size_t i = tid; i < N; i += stride) {
    output[i] = input[i] * 2.0f;
}
```

## Key APIs

### From `cuda.core`:

- `Device` - Device management and context
- `Program` - Compile CUDA C++ kernels
- `ProgramOptions` - Kernel compilation options (architecture target)
- `LaunchConfig` - Configure grid/block dimensions and shared memory
- `launch()` - Execute kernel
- `EventOptions` - GPU timing configuration

### From CuPy:

- `cp.asarray()` - Transfer data to GPU
- `cp.zeros_like()` - Allocate GPU arrays

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
python blockwiseSum.py
```

## Expected Output

```
Device: <Your GPU>
Compute Capability: sm_XX
Array size: 1,048,576 elements

Simple indexing: Test PASSED
Strided loop:    Test PASSED
Block-wise sum:  Test PASSED

Kernel time: X.XXX ms, Bandwidth: XXX.X GB/s

Done
```

## Files

- `blockwiseSum.py` - Python implementation with CUDA kernels
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [cuda.core Documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [CUDA Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [CuPy Documentation](https://docs.cupy.dev/)
