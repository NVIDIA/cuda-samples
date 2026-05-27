# Sample: Numpy vs. Cupy (Python)

## Description

This sample demonstrates performance comparison between NumPy (CPU) and CuPy (GPU) for matrix multiplication operations. It benchmarks the execution time of matrix dot products on both CPU and GPU, showing the performance benefits of GPU acceleration for numerical computations.

## What you will learn

- How to set up and use CuPy for GPU-accelerated numerical computations.
- How to benchmark NumPy vs CuPy performance for matrix operations.
- How to transfer data between CPU (NumPy) and GPU (CuPy) memory using `cp.asarray()`.
- How to use CUDA device management with the cuda-core library.
- How to validate computational results between CPU and GPU implementations using `np.testing.assert_allclose()`.
- How to handle GPU warmup to avoid first-run overhead in benchmarking.
- How to create and manage explicit CUDA streams with `device.create_stream()`.
- How to properly cleanup streams with `stream.close()` in try/finally blocks.
- How to access GPU device information (name, compute capability).
- How to create timing context managers for performance measurement using CUDA events.

## Key libraries

- `numpy`
- `cupy`
- `cuda-core`

## Key APIs

**From cuda.core:**
- `Device()` – Get CUDA device object for specific GPU
- `device.create_stream()` – Create explicit CUDA stream
- `stream.close()` – Close and cleanup stream resources

## Requirements
1. **NVIDIA Graphics Card** with CUDA support
2. **CUDA Drivers** installed on your system
3. **CUDA Toolkit** installed on your system
4. **Python 3.12 or newer**

**Install packages:**
```bash
pip install -r requirements.txt
```

## How to run

Basic usage:
```bash
# Pre-steps:
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Run from the Python directory:
cd /path/to/numpyVsCupy/Python
python -m 1_GettingStarted.numpyVsCupy.numpyVsCupy
```

With custom parameters:
```bash
python -m 1_GettingStarted.numpyVsCupy.numpyVsCupy --n_size 5000
```

### Command line arguments

- `--n_size`, `-n`: Size of the matrix (n * n) for benchmarking (default: 4096)

## Expected Output
```
Validation PASSED: NumPy and CuPy results match within tolerance
Demo completed successfully!
```

## Files
- `numpyVsCupy.py` – Python implementation
- `README.md` – This file
- `requirements.txt` – Required packages
