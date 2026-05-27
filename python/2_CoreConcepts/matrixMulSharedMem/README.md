# Matrix Multiplication with Shared Memory (GEMM)

Demonstrates efficient matrix multiplication using nvmath-python APIs and custom CUDA kernels with tiling, shared memory, and loop unrolling.

## Overview

- Uses nvmath.linalg.advanced.Matmul for high-performance GEMM via cuBLASLt
- Compares with custom CUDA kernel using tiling and shared memory
- Shows how tiling reduces global memory bandwidth requirements
- Demonstrates shared memory for data reuse within thread blocks
- Uses loop unrolling to improve instruction-level parallelism

## What You'll Learn

- How to use nvmath stateful API for optimized matrix multiplication
- How to tile matrix operations for better cache locality
- Using shared memory to reduce redundant global memory accesses
- Loop unrolling techniques for GPU kernels
- Benchmarking and comparing kernel performance

## Key Libraries

- `nvmath-python` - NVIDIA math library with cuBLASLt access
- `cuda.core` - Modern CUDA Python API for custom kernel compilation
- `cupy` - GPU array library for Python

## Key APIs

### From `nvmath.linalg.advanced`:

- `Matmul()` - Stateful matrix multiplication with planning and execution phases
- `MatmulComputeType` - Compute type options for mixed-precision

### From `cuda.core`:

- `Device()` - CUDA device management and properties
- `Program()` - Runtime kernel compilation (NVRTC)
- `LaunchConfig()` - Kernel launch configuration (grid/block dimensions)
- `launch()` - Kernel execution on a stream
- `Stream.record_event()` / `Event.elapsed_time()` - GPU timing

## Requirements

### Hardware:

- NVIDIA GPU with Compute Capability 7.0 or higher
- Minimum GPU memory: 256 MB (for 1024×1024 matrices)

### Software:

- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- See requirements.txt for package dependencies

## Installation

```bash
cd cuda-samples/python/2_CoreConcepts/matrixMulSharedMem
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to Run

```bash
python matrixMulSharedMem.py
```

## Expected Output

```
======================================================================
Matrix Multiplication with Shared Memory (GEMM)
Using nvmath and cuda.core APIs
======================================================================

Device: NVIDIA GeForce RTX 4090
Compute Capability: sm_89

Custom kernel compiled ✓

Matrix dimensions: A(1024x1024) × B(1024x1024) = C(1024x1024)
Custom kernel tile size: 16x16

----------------------------------------------------------------------
NVMATH MATMUL (cuBLASLt)
----------------------------------------------------------------------
Using nvmath.linalg.advanced.Matmul stateful API
Average time: X.XXX ms
Performance: XXXX.XX GFLOPS

----------------------------------------------------------------------
CUSTOM KERNEL (Tiled + Shared Memory + Loop Unrolling)
----------------------------------------------------------------------
Grid: (64, 64), Block: (16, 16)
Average time: X.XXX ms
Performance: XXX.XX GFLOPS

----------------------------------------------------------------------
VERIFICATION
----------------------------------------------------------------------
nvmath         : PASSED (max error: X.XXe-XX)
Custom kernel  : PASSED (max error: X.XXe-XX)

======================================================================
PERFORMANCE SUMMARY
======================================================================
Implementation                 Time (ms)    GFLOPS
----------------------------------------------------------------------
nvmath (cuBLASLt)              X.XXX        XXXX.XX
Custom (shared mem + unroll)   X.XXX        XXX.XX
```

## Tiling Concept

```
     Matrix A (M×K)          Matrix B (K×N)          Matrix C (M×N)
    ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
    │ T00 │ T01 │...│       │ T00 │ T01 │...│       │     │     │   │
    ├─────┼─────┼───┤       ├─────┼─────┼───┤       ├─────┼─────┼───┤
    │ T10 │ T11 │...│   ×   │ T10 │ T11 │...│   =   │     │ Cij │   │
    ├─────┼─────┼───┤       ├─────┼─────┼───┤       ├─────┼─────┼───┤
    │ ... │ ... │...│       │ ... │ ... │...│       │     │     │   │
    └───────────────┘       └───────────────┘       └───────────────┘

    Cij = Σ (A_tile_row × B_tile_col) for all tiles along K
```

## nvmath Stateful API

```python
import nvmath.linalg.advanced as nvmath_advanced

# Create matrices (CuPy arrays)
A = cp.random.rand(m, k).astype(cp.float32)
B = cp.random.rand(k, n).astype(cp.float32)

# Use stateful API for fine-grained control
with nvmath_advanced.Matmul(A, B) as mm:
    mm.plan()           # Find optimal algorithm
    C = mm.execute()    # Execute computation
```

## Memory Access Optimization (Custom Kernel)

| Implementation | Global Reads per C element | Reduction |
|---------------|---------------------------|-----------|
| Naive         | 2 × K                     | (baseline)|
| Tiled (16×16) | 2 × K / 16                | 16×       |

## Files

- `matrixMulSharedMem.py` - Python implementation comparing nvmath vs custom kernel
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [nvmath-python Documentation](https://docs.nvidia.com/cuda/nvmath-python/)
- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [cuda.core API Guide](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [CuPy Documentation](https://docs.cupy.dev/)
