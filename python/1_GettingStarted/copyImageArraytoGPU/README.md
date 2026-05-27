# Sample: Image Array Copy to GPU (Python)

## Description

Copy image arrays between CPU and GPU memory using the modern `cuda.core` API with optimal performance through pinned memory and asynchronous transfers.

## What You'll Learn

- How to use pinned memory for faster CPU↔GPU transfers
- Using the `cuda.core` API for memory management
- Working with DLPack for zero-copy array views
- Performing asynchronous memory transfers with CUDA streams
- Interoperability between CUDA Core API and CuPy
- Proper CUDA resource management and cleanup

## Key Libraries

- `cuda.core` - Modern CUDA Python API
- `numpy` - Array operations and DLPack support
- `cupy` - GPU array operations and CUDA interoperability

## Key APIs

### From `cuda.core`:

- `Device()` - Initialize and access CUDA device
- `Device.set_current()` - Set the current device for API calls
- `Device.create_stream()` - Create CUDA stream for async operations
- `Device.memory_resource` - Access device memory allocator
- `PinnedMemoryResource()` - Allocate pinned host memory
- `buffer.copy_to()` - Copy data between memory spaces
- `buffer.close()` - Release allocated memory

### From `numpy`:

- `np.from_dlpack()` - Create array view from DLPack capsule
- `np.copyto()` - Copy data between arrays

### From `cupy`:

- `cp.from_dlpack()` - Create GPU array view from DLPack capsule
- `cp.cuda.ExternalStream()` - Use external CUDA stream

### From `cuda_samples_utils`:

- `verify_array_result()` - Verify computation results

## Requirements

### Hardware:

- NVIDIA GPU with CUDA support
- Sufficient GPU memory for image data (sample uses ~200KB for 256×256×3 image)

### Software:

- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- NumPy 2.3.2 or newer (required for DLPack support)
- `cuda-python` package (>=13.0.0+)
- `cuda-core` package (>=0.6.0)
- `cupy-cuda13x` package (13.0.0+)

## Installation

Install the required packages from requirements.txt:

```bash
cd /path/to/cuda-samples/python/1_GettingStarted/copyImageArraytoGPU
pip install -r requirements.txt
```

The requirements.txt installs:
- `numpy` (2.3.2+, required for DLPack)
- `cuda-python` (>=13.0.0+)
- `cuda-core` (>=0.6.0)
- `cupy-cuda13x` (13.0.0+)

## How to Run

### Basic usage:

```bash
cd samples/python/1_GettingStarted/copyImageArraytoGPU
python copyImageArraytoGPU.py
```

## Expected Output

```
[Image Array Copy to GPU using CUDA Core API]
Device: NVIDIA GeForce RTX 4090
[Image array copy of 256x256x3 image]
Creating sample image...
Copying image to GPU...
Creating CuPy view of GPU data...
Mean pixel value (computed on GPU): 127.50
Copying image back from GPU...
Verifying result...
Test PASSED

Done
```

**Note:** Device name will vary based on your GPU.

## Files

- `copyImageArraytoGPU.py` - Python implementation using cuda.core API
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_samples_utils.py` - Common utilities (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [cuda.core API Guide](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [DLPack Specification](https://dmlc.github.io/dlpack/latest/)
- [CuPy Documentation](https://docs.cupy.dev/)
