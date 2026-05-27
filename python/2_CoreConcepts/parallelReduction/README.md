# Sample: Parallel Reduction (Python)

## Description

Efficiently sum a large array on GPU using parallel reduction. This sample demonstrates:
1. **Custom CUDA kernel** showing reduction tree pattern and synchronization
2. **cuda.compute.reduce_into()** for production-ready reduction

## What You'll Learn

- **Reduction tree pattern**: Divide-and-conquer parallel algorithm
- **Thread synchronization**: Using `__syncthreads()` for coordination
- **Avoiding warp divergence**: Sequential thread IDs vs strided IDs

## Key Concepts

### Reduction Tree Pattern

Parallel reduction uses a tree-based approach where each iteration halves active elements:

```
Initial:  [a0, a1, a2, a3, a4, a5, a6, a7]
Step 1:   [a0+a4, a1+a5, a2+a6, a3+a7]      threads 0-3 active
Step 2:   [a0+a2+a4+a6, a1+a3+a5+a7]        threads 0-1 active
Step 3:   [sum of all]                       thread 0 only
```

This requires only `log2(N)` steps to reduce N elements.

### Avoiding Warp Divergence

```c
// Good: Sequential thread IDs (warps stay coherent)
if (tid < s) {
    sdata[tid] += sdata[tid + s];
}

// Bad: Strided IDs (causes warp divergence)
if (tid % (2 * s) == 0) {  // Don't do this!
    sdata[tid] += sdata[tid + s];
}
```

## Requirements

### Hardware

- NVIDIA GPU with CUDA support

### Software

- CUDA Toolkit 13.0+
- Python 3.10+
- `cuda-python` (13.0.0+)
- `cuda-core` (>=0.6.0)
- `cuda-cccl` (1.0.0+)
- `cupy` (13.0.0+)
- `numpy` (>=2.3.2)

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python parallelReduction.py
```

## Expected Output

```
======================================================================
Parallel Reduction - Efficient GPU Array Summation
======================================================================

Device: <your GPU name>
Compute Capability: <version>

Array size: 1,048,576 elements (4.2 MB)
Expected sum: <value>

Compiling custom CUDA kernel...

======================================================================
PART 1: Custom Kernel (Educational)
======================================================================

Reduction tree kernel:       <result>
Expected:                    <result>
Error:                       <small value>
Time:                        <varies> ms

======================================================================
PART 2: cuda.compute.reduce_into() (Production)
======================================================================

cuda.compute result:         <result>
Expected:                    <result>
Error:                       <small value>
Time:                        <varies> ms

Test PASSED!
```

Note: Exact values vary due to random input data. `cuda.compute.reduce_into()` is typically faster than the custom kernel because it calls CUB's `DeviceReduce`, which uses highly tuned, architecture‑specific kernels and optimized memory access patterns.

## Files

- `parallelReduction.py` - Custom kernel + cuda.compute comparison
- `README.md` - This documentation
- `requirements.txt` - Python dependencies

## See Also

- [Mark Harris - Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [cuda.core Documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/)
