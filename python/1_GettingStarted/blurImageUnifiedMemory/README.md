# Sample: Image Blur with Unified Memory (Python)

## Description

Blur images on GPU using modern `cuda.core` APIs for kernel compilation, execution, and memory management. This sample demonstrates **zero-copy data sharing** between CPU and GPU using unified (managed) memory.

## What You'll Learn

- Compiling CUDA kernels at runtime with `cuda.core.Program`
- Launching kernels with `cuda.core.launch` and `LaunchConfig`
- Using unified memory with `cuda.core.ManagedMemoryResource`
- **Zero-copy CPU access** to unified memory via `np.from_dlpack()`
- Seamless CPU/GPU memory access without explicit transfers

## Key Concepts

### Kernel Compilation with cuda.core.Program

```python
# Compile CUDA C++ kernel at runtime
program = Program(KERNEL_CODE, code_type="c++", options=options)
compiled = program.compile(target_type="cubin")
kernel = compiled.get_kernel("box_blur_3x3")
```

### Kernel Launch with cuda.core.launch

```python
# Configure and launch kernel
config = LaunchConfig(grid=grid_size, block=block_size)

# Buffers can be passed directly as kernel arguments
launch(stream, config, kernel, src_buf, dst_buf, H, W)
```

### Unified Memory (Managed Memory)

This sample uses `ManagedMemoryResource` for simplicity: a single allocation is accessible from both CPU and GPU without explicit transfers. For performance-critical workloads, consider `LegacyPinnedMemoryResource` + `DeviceMemoryResource` instead, which gives explicit control over host/device placement and transfer costs.

Unified memory is accessible from both CPU and GPU without explicit data transfers:

```python
# Allocate unified memory
options = ManagedMemoryResourceOptions(preferred_location=device.device_id)
mr = ManagedMemoryResource(options)
src_buf = mr.allocate(n_bytes, stream)
dst_buf = mr.allocate(n_bytes, stream)
try:
    # Synchronize to ensure allocations are complete before CPU access
    stream.sync()

    # Create numpy views of unified memory using DLPack protocol (zero-copy)
    src_np = np.from_dlpack(src_buf).view(np.float32).reshape(H, W)
    dst_np = np.from_dlpack(dst_buf).view(np.float32).reshape(H, W)

    # CPU writes directly to unified memory
    src_np[:] = input_data

    # Launch kernel - buffers can be passed directly as arguments
    launch(stream, config, kernel, src_buf, dst_buf, H, W)
    stream.sync()

    # Return zero-copy view; caller must close buffers when done
    return dst_np, src_buf, dst_buf
except Exception:
    src_buf.close()
    dst_buf.close()
    raise
```

When returning a zero-copy view, the caller must close the buffers after use (e.g., in a `try/finally` block) to avoid leaking managed memory.

## Key APIs

### From `cuda.core`:

- `Device` - CUDA device management
- `Program` - Runtime kernel compilation (NVRTC)
- `ProgramOptions` - Compilation options (architecture target)
- `LaunchConfig` - Kernel launch configuration (grid/block dimensions)
- `launch` - Execute compiled kernel
- `ManagedMemoryResource` - Unified memory allocation

### Zero-Copy Techniques:

- `np.from_dlpack(buffer)` - Create numpy view of unified memory using DLPack protocol
- Pass `buffer` directly to `launch()` as kernel arguments
- When returning a zero-copy view, return `(view, src_buf, dst_buf)` and have the caller close buffers in `try/finally` after use

## Kernel Techniques

- **2D Thread Mapping** - Each thread computes one output pixel
- **Stencil Pattern** - Read neighboring pixels (3x3 neighborhood)
- **Boundary Handling** - Clamp to edge for border pixels
- **Box Filter** - 3x3 averaging for blur effect

## Requirements

### Hardware:

- NVIDIA GPU with CUDA support
- Minimum GPU memory: 256 MB

### Software:

- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- `cuda-python` package (13.0.0+)
- `cuda-core` package (>=0.6.0)
- `numpy` package (>=2.3.2)
- `pillow` package (10.0.0+)

## Installation

```bash
cd /path/to/cuda-samples/python/1_GettingStarted/blurImageUnifiedMemory
pip install -r requirements.txt
```

## How to Run

```bash
python blurImageUnifiedMemory.py
```

## Expected Output

```
============================================================
Image Blur with Unified Memory (cuda.core)
============================================================

Device: <Your GPU Name>
Compute Capability: sm_<XY>

Compiling CUDA kernel with cuda.core.Program...
  Compiled for architecture: sm_<XY>

Image size: 256x256 grayscale
Creating sample image...
Blurring image on GPU...

Saving results...
  Saved: original_image.png
  Saved: blurred_image.png

Verifying result...
  Test PASSED
  Max difference from original: <value>
```

## Output Files

- `original_image.png` - Test pattern image before blur
- `blurred_image.png` - Image after 3x3 box blur

## Files

- `blurImageUnifiedMemory.py` - Python implementation using cuda.core
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [cuda.core Documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [cuda.core.Program](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Program.html)
- [cuda.core.ManagedMemoryResource](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.ManagedMemoryResource.html)
- [CUDA Managed Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)
