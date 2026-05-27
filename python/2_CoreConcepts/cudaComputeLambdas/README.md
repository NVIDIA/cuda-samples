# cudaComputeLambdas (Python)

## Description

This sample demonstrates how **cuda.compute** (from the
`cuda-cccl` package) accepts plain Python callables, including
lambdas, as the operators that drive device-wide reductions,
transforms, and scans. Internally `cuda.compute` JIT-compiles the
callable through Numba for the GPU, so you can iterate on the
operator in pure Python and still get a fused device-wide kernel.

The sample exercises three algorithm families:

1. `cuda.compute.reduce_into` - sum via `lambda a, b: a + b`.
2. `cuda.compute.unary_transform` - elementwise `y = x*x + 1` via a
   lambda.
3. `cuda.compute.inclusive_scan` - prefix sum over only the even
   values, driven by a regular Python function as the binary
   operator.

## What You'll Learn

- Passing a Python `lambda` directly as the operator to a cuda.compute
  device algorithm
- Using a regular Python `def` function for the same purpose when the
  op is non-trivial
- The three core algorithm families in cuda.compute: reductions,
  transforms, and scans
- How cuda.compute auto-compiles the op to LTO-IR via Numba

## Key Libraries

- [`cuda.compute`](https://nvidia.github.io/cccl/python.html) (from the `cuda-cccl` package) - device algorithms and JIT-compiled Python ops
- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - device setup
- `cupy` - device buffers
- `numpy` - scalar init values and host-side verification

## Key APIs

### From `cuda.compute`

- `cuda.compute.reduce_into(d_in, d_out, num_items, op, h_init)` - device-wide reduction
- `cuda.compute.unary_transform(d_in, d_out, num_items, op)` - elementwise unary transform
- `cuda.compute.inclusive_scan(d_in, d_out, op, init_value, num_items)` - inclusive prefix scan

### From `cuda_samples_utils`

- `print_gpu_info()` - print device name and compute capability

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.0 or higher

### Software

- CUDA Toolkit 13.0 or newer (cuda.compute compiles ops to LTO-IR via
  Numba, which needs the toolkit's `nvvm` and `libdevice`).
- Python 3.10 or newer
- `cuda-cccl` (>=1.0.0)
- `cuda-core` (>=1.0.0)
- `cupy-cuda13x` (>=14.0.0)
- `numba-cuda` (pulled in transitively by `cuda-cccl`)

If the CUDA toolkit is not on your `PATH`, set `CUDA_HOME` so Numba
can locate `libdevice`:

```bash
export CUDA_HOME=/usr/local/cuda
```

## Installation

Install the required packages from `requirements.txt`:

```bash
cd /path/to/cuda-samples/python/2_CoreConcepts/cudaComputeLambdas
pip install -r requirements.txt
```

The `requirements.txt` installs:

- `cuda-cccl` (>=1.0.0) - ships the `cuda.compute` module
- `cuda-core` (>=1.0.0)
- `cupy-cuda13x` (>=14.0.0)
- `numpy` (>=1.24.0)

## How to Run

### Basic usage

```bash
cd cuda-samples/python/2_CoreConcepts/cudaComputeLambdas
python cudaComputeLambdas.py
```

### With custom parameters

```bash
python cudaComputeLambdas.py --device 1
```

## Expected Output

```
Device: <Your GPU Name>
Compute Capability: <X.Y>

reduce_into(lambda a,b: a+b) over 1..10 -> 55 (expected 55)  OK

unary_transform(lambda x: x*x + 1):
  got      = [1, 2, 5, 10, 17, 26, 37, 50]
  expected = [1, 2, 5, 10, 17, 26, 37, 50]  OK

inclusive_scan(add-evens-only) over [1,2,3,4,5,6]:
  got      = [0, 2, 2, 6, 6, 12]
  expected = [0, 2, 2, 6, 6, 12]  OK

Done
```

**Note:** Device name and compute capability will vary based on your GPU.

## Files

- `cudaComputeLambdas.py` - Python implementation
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_samples_utils.py` - Common utilities (imported by this sample)
