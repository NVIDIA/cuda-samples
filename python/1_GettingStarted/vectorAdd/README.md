# Sample: Vector Addition (Python)

## Description

Run your first GPU kernel: add two vectors element-wise on the GPU using the [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) API with runtime compilation.

## What You'll Learn

- Writing CUDA kernels in C++ with template support
- Runtime compilation of CUDA kernels from Python
- Using `cuda.core` for device management, programs, and launches
- Configuring and launching kernels with grid and block dimensions
- Using CuPy for GPU memory management
- Verifying GPU results against CPU computation

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) — Pythonic access to CUDA runtime and compilation
- `cupy` — GPU array library for Python

## Key APIs

### From `cuda.core`

- `Device` — Initialize and manage CUDA device
- `Program` — Create program from kernel source code
- `ProgramOptions` — Set compilation options (C++ standard, architecture)
- `LaunchConfig` — Configure kernel launch parameters
- `launch` — Execute kernel on specified stream

Import stable symbols from the top-level package (not `cuda.core.experimental`). See the [cuda.core documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/).

### From CuPy

- `cp.random.rand()` — Generate random arrays on GPU
- `cp.empty()` — Allocate uninitialized GPU arrays
- `cp.allclose()` — Verify results with tolerance

### From `cuda_samples_utils`

- `verify_array_result()` — Verify computation results

## Kernel Techniques

- **1D Grid-Stride Loop** — Handle arbitrary array sizes with fixed grid
- **Template Programming** — Generic kernel for different data types
- **Bounds Checking** — Prevent out-of-bounds memory access

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.0 or higher
- Minimum GPU memory: 512 MB

### Software

- CUDA Toolkit 13.0 or newer (matches `cuda-python` 13.x)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=0.6.0)
- `cupy-cuda13x` (>=13.0.0)

## Installation

Install the required packages from requirements.txt:

```bash
cd /path/to/cuda-samples/python/1_GettingStarted/vectorAdd
pip install -r requirements.txt
```

The requirements.txt installs:

- `cuda-python` (>=13.0.0)
- `cuda-core` (>=0.6.0)
- `cupy-cuda13x` (>=13.0.0)

## How to Run

### Basic usage

```bash
cd samples/python/1_GettingStarted/vectorAdd
python vectorAdd.py
```

### With custom parameters

```bash
# Custom vector size
python vectorAdd.py --elements 1000000

# Use specific GPU
python vectorAdd.py --device 1

# Skip verification for benchmarking
python vectorAdd.py --no-verify
```

## Expected Output

```
[Vector addition using CUDA Core API]
Device: <Your GPU Name>
Compute Capability: sm_<XX>
Compiling kernel 'vectorAdd<float>'...
Kernel compiled successfully
[Vector addition of 50000 elements]
CUDA kernel launch with 196 blocks of 256 threads
Verifying result...
Test PASSED

Done
```

**Note:** Device name and compute capability will vary based on your GPU.

## Files

- `vectorAdd.py` — Python implementation using cuda.core API
- `README.md` — This file
- `requirements.txt` — Sample dependencies
- `../../Utilities/cuda_samples_utils.py` — Common utilities (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [cuda.core API](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [CuPy Documentation](https://docs.cupy.dev/)
