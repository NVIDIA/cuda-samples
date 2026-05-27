# Sample: Memory Resources and Buffers (Python)

## Description

This sample demonstrates the `cuda.core` memory management model: a
`MemoryResource` owns a pool of memory and hands out `Buffer` objects that
can be passed to kernels, copied between resources with
`Buffer.copy_to()`, and viewed as NumPy or CuPy arrays through DLPack. The
script exercises three common resources side-by-side:

1. **`DeviceMemoryResource`** - device-local GPU memory. Every `Device`
   exposes a default pool via `Device.memory_resource`, and applications
   can create additional pools explicitly.
2. **`PinnedMemoryResource`** - page-locked host memory, used here as the
   input and output staging buffers around a GPU kernel (the canonical
   pinned-H2D / compute / pinned-D2H pattern).
3. **`ManagedMemoryResource`** - unified memory that the driver migrates
   between host and device on demand; host views see the GPU's writes
   without an explicit copy.

The same `scale_and_bias` kernel runs on each resource and every result is
verified on the host.

## What You'll Learn

- Creating and using `DeviceMemoryResource`, `PinnedMemoryResource`, and
  `ManagedMemoryResource`
- Allocating `Buffer` objects from a resource with a bound stream
- Copying between buffers across resources with `Buffer.copy_to()`
- Taking zero-copy NumPy or CuPy views of a `Buffer` via DLPack
- Releasing buffers with stream-ordered `close(stream)` semantics

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - Pythonic access to CUDA runtime, programs, and memory resources
- `cupy` - GPU array views of device buffers
- `numpy` - host array views of pinned and managed buffers

## Key APIs

### From `cuda.core`

- `Device.memory_resource` - default memory pool attached to a device
- `DeviceMemoryResource`, `PinnedMemoryResource`, `ManagedMemoryResource` - allocate buffers of the corresponding memory kind
- `MemoryResource.allocate(nbytes, stream=...)` - returns a `Buffer`
- `Buffer.copy_to(dst_buffer, stream=...)` - async, stream-ordered copy
- `Buffer.close(stream)` - stream-ordered deallocation
- `Buffer` supports `__dlpack__` for zero-copy views

### From CuPy and NumPy

- `cp.from_dlpack()` / `np.from_dlpack()` - zero-copy array view of a `Buffer`

### From `cuda_samples_utils`

- `print_gpu_info()` - print device name and compute capability

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.0 or higher
- Managed memory support (most discrete GPUs on Linux and Windows)

### Software

- CUDA Toolkit 13.0 or newer (matches `cuda-python` 13.x)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=0.6.0)
- `cupy-cuda13x` (>=13.0.0)

## Installation

Install the required packages from `requirements.txt`:

```bash
cd /path/to/cuda-samples/python/2_CoreConcepts/memoryResources
pip install -r requirements.txt
```

The `requirements.txt` installs:

- `cuda-python` (>=13.0.0)
- `cuda-core` (>=0.6.0)
- `cupy-cuda13x` (>=13.0.0)

## How to Run

### Basic usage

```bash
cd cuda-samples/python/2_CoreConcepts/memoryResources
python memoryResources.py
```

### With custom parameters

```bash
# Larger buffer size
python memoryResources.py --elements 1048576

# Use a specific GPU
python memoryResources.py --device 1
```

## Expected Output

```
Device: <Your GPU Name>
Compute Capability: <X.Y>

[1] DeviceMemoryResource + PinnedMemoryResource (staging)
  Pinned staging, device kernel, and copy_to verified

[2] ManagedMemoryResource (unified memory)
  GPU writes observed directly through the host-visible mapping

[3] Explicit DeviceMemoryResource
  Explicit DeviceMemoryResource allocation verified

All memory resource demos passed.
```

**Note:** Device name and compute capability will vary based on your GPU.

## Files

- `memoryResources.py` - Python implementation using `cuda.core` memory resources
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_samples_utils.py` - Common utilities (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.core` memory API](https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#memory-management)
- Upstream `cuda.core` example: [`memory_ops.py`](https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/memory_ops.py)
- Upstream `cuda.core` example: [`memory_pool_resources.py`](https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/memory_pool_resources.py)
