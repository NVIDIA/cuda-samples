# tmaTensorMap (Python)

## Description

This sample demonstrates how to use Tensor Memory Accelerator (TMA)
descriptors with `cuda.core` on Hopper and later GPUs (compute
capability >= 9.0). TMA enables efficient bulk data movement between
global and shared memory using hardware-managed tensor map
descriptors, which are a key building block for modern GEMM kernels
and large shared-memory tile loads.

The sample:

1. Creates a TMA tiled descriptor from a CuPy device array via
   `StridedMemoryView.from_any_interface(...).as_tensor_map(...)`.
2. Passes the descriptor by value (as `__grid_constant__`) to a
   kernel that uses libcudacxx TMA/barrier wrappers to bulk-load a
   tile into shared memory, then copies it out to verify correctness.
3. Reuses the same descriptor against a new source tensor with
   `replace_address()` to avoid rebuilding it.

## What You'll Learn

- Creating a TMA descriptor from a strided device tensor via
  `StridedMemoryView.as_tensor_map(box_dim=...)`
- Passing a tensor map to a kernel by value using
  `__grid_constant__`
- Using libcudacxx (`cuda/barrier`) to coordinate TMA loads with a
  block-scoped barrier
- Reusing a descriptor against a new source buffer via
  `tensor_map.replace_address(new_tensor)`
- Compiling a kernel to CUBIN for a specific target arch so Hopper
  features are available
- Using `cuda.pathfinder` to locate the CUDA toolkit include directory
  CCCL headers and libcudacxx

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - compilation, launching, and tensor-map helpers
- `cuda.pathfinder` - locate the CUDA toolkit include directory
- `cupy` - allocate and fill device tensors
- `numpy` - scalar kernel arguments

## Key APIs

### From `cuda.core`

- `StridedMemoryView.from_any_interface(tensor, stream_ptr=-1)` - build a typed view from any DLPack/CUDA-array-interface tensor
- `StridedMemoryView.as_tensor_map(box_dim=(...))` - produce a TMA descriptor for the given tile shape
- `tensor_map.replace_address(new_tensor)` - retarget an existing descriptor at a new tensor
- `Program(code, code_type="c++", options=ProgramOptions(std="c++17", arch="sm_90", include_path=[...]))` - compile a C++ kernel against libcudacxx
- `program.compile("cubin")` - produce a CUBIN so `__grid_constant__` and TMA intrinsics are fully supported
- `launch(stream, config, kernel, tensor_map, ...)` - pass the TMA descriptor as a kernel argument

### From `cuda.pathfinder`

- `get_cuda_path_or_home()` - return the detected CUDA toolkit root for locating `include/cccl`

### From `cuda_samples_utils`

- `print_gpu_info()` - print device name and compute capability

## Requirements

### Hardware

- NVIDIA Hopper or newer GPU with Compute Capability 9.0 or higher (H100, H200, B200, ...)
- On GPUs older than Hopper the sample exits cleanly without running the kernel
- Minimum GPU memory: 512 MB

### Software

- CUDA Toolkit 13.0 or newer with libcudacxx (cccl) headers
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=0.6.0)
- `cupy-cuda13x` (>=13.0.0)

## Installation

Install the required packages from `requirements.txt`:

```bash
cd /path/to/cuda-samples/python/2_CoreConcepts/tmaTensorMap
pip install -r requirements.txt
```

The `requirements.txt` installs:

- `cuda-python` (>=13.0.0)
- `cuda-core` (>=0.6.0)
- `cupy-cuda13x` (>=13.0.0)

## How to Run

### Basic usage

```bash
cd cuda-samples/python/2_CoreConcepts/tmaTensorMap
python tmaTensorMap.py
```

### With custom parameters

```bash
# Larger tensor (must be a multiple of the 128-element tile)
python tmaTensorMap.py --elements 8192

# Use a specific GPU
python tmaTensorMap.py --device 1
```

## Expected Output

On a Hopper (sm_90) GPU:

```
Device: NVIDIA H100 PCIe
Compute Capability: 9.0

TMA copy verified: 1024 elements across 8 tiles
replace_address verified: descriptor reused with new source tensor
```

**Note:** Device name and compute capability will vary based on your GPU.

## Files

- `tmaTensorMap.py` - Python implementation using `cuda.core` TMA APIs
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_samples_utils.py` - Common utilities (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [TMA in the CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-accelerator)
- [`cuda::barrier` reference](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/barrier.html)
