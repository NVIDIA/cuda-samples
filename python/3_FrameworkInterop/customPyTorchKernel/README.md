# Sample: PyTorch Custom GPU Operator

## Description

This sample demonstrates how to add a custom GPU operation to PyTorch using the `cuda.core` API. It implements a simple square operation (y = x²) to show the complete workflow from CUDA kernel to PyTorch integration with autograd support.

## Requirements

- NVIDIA GPU with Compute Capability 7.0+
- CUDA Toolkit 13.0+
- Python 3.10+
- PyTorch 2.0+
- cuda-python >= 13.0.0
- cuda-core >=1.0.0

## Installation

```bash
cd python/3_FrameworkInterop/customPyTorchKernel
pip install -r requirements.txt
```

**Windows users:** The default `torch` wheel on PyPI for Windows is CPU-only and will cause `torch.cuda.is_available()` to return `False`. Install a CUDA-enabled build from PyTorch's wheel index *before* (or after) the command above:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Replace `cu128` with the wheel suffix matching your installed CUDA driver (e.g. `cu121`, `cu124`, `cu126`, `cu128`). The driver's CUDA version must be >= the wheel's bundled runtime.

## How to Run

```bash
# Basic usage
python customPyTorchKernel.py

# Test with more elements
python customPyTorchKernel.py --size 1000000

# Use specific GPU
CUDA_VISIBLE_DEVICES=1 python customPyTorchKernel.py
```

## Expected Output

The sample runs three tests:
1. Forward pass correctness (y = x²)
2. Backward pass correctness (gradient computation)
3. Multi-dimensional tensor support

All tests should pass, confirming the custom operator works correctly with PyTorch's autograd system.

## Key Concepts

The sample demonstrates:
- Writing CUDA kernels with grid-stride loops
- Runtime kernel compilation with cuda.core
- PyTorch autograd integration via `torch.autograd.Function`
- Stream management using PyTorch's current stream
- Kernel caching for performance

The code is self-documenting with inline comments explaining each step.
