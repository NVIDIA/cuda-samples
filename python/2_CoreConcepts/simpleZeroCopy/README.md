# Sample: simpleZeroCopy (Python)

## Description

This sample demonstrates zero-copy access using **`cuda.core`** to compile and launch a kernel, and **`cuda.bindings.runtime`** for mapped pinned host memory (`cudaHostAlloc` with `cudaHostAllocMapped`, `cudaHostGetDevicePointer`, and `cudaFreeHost`). The GPU loads and stores through **device** addresses that refer to that host memory—no `cudaMemcpy` in or out. The example is vector add with inputs and output as NumPy views of the host side of those buffers.

## What you will learn

- How to allocate **mapped** pinned host memory with `cudaHostAlloc` (via `cuda.bindings.runtime`) so the GPU can use `cudaHostGetDevicePointer` addresses in a kernel
- How `cuda.core.PinnedMemoryResource` differs (staging/copies; not guaranteed to be `cudaHostAllocMapped` for direct kernel access)
- How to build NumPy views of host addresses with `ctypes` and `numpy.frombuffer`
- How to launch CUDA kernels with `cuda.core`’s `Program` and `launch`, passing **device** pointers for mapped buffers
- When zero-copy is beneficial vs. device memory with explicit transfers
- How to validate results on the host without a D2H memcpy

## Key libraries

- `numpy` – CPU arrays and reference computation
- `cuda-core` – `Device`, stream, `Program`, `LaunchConfig`, `launch`
- `cuda-python` (`cuda.bindings.runtime`) – `cudaHostAlloc` / `cudaHostGetDevicePointer` / `cudaFreeHost` for mapped host memory

## Key APIs

**From cuda.core:** `Device`, `device.create_stream()`, `Program`, `ProgramOptions`, `LaunchConfig`, `launch`

**From cuda.bindings.runtime:** `cudaHostAlloc` (with `cudaHostAllocMapped` | `cudaHostAllocPortable`), `cudaHostGetDevicePointer`, `cudaFreeHost`

**From the standard library:** `ctypes` – wrap host pointers for `numpy.frombuffer` float32 views

**Memory management:** Free host memory with `cudaFreeHost` in a `finally` block; call `stream.close()` when done.

## Zero-Copy Memory: When to Use

### Benefits
- **No explicit transfers**: Simplifies code by eliminating `cudaMemcpy` calls
- **Automatic synchronization**: Host can access results immediately after kernel completes
- **Good for small data**: Overhead of explicit transfers can exceed benefits for small arrays
- **Excellent for integrated GPUs**: On systems like Jetson (Tegra), CPU and GPU share physical memory

### Limitations
- **Slower access**: Limited by PCIe bandwidth vs. device memory bandwidth
- **Not for compute-intensive**: Device memory is much faster for frequently accessed data
- **Discrete GPU overhead**: Each access crosses PCIe bus

### Best Use Cases
1. Small data sets where transfer overhead dominates
2. Data accessed infrequently by GPU
3. Integrated GPU platforms (shared memory)
4. Streaming data from host to device
5. Prototyping and debugging (simplifies memory management)

## Requirements

1. **NVIDIA GPU** and a **driver** compatible with your installed `cuda-python` / `cuda-core` wheels.
2. **Python 3.10 or newer**
3. Install **`pip install -r requirements.txt`** (NumPy, `cuda-python`, `cuda-core`). A **system** CUDA Toolkit is not strictly required if the process can load the driver/runtime; use `LD_LIBRARY_PATH` in *How to run* if you hit missing-library errors.

**Install packages:**
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy>=2.3.2 cuda-core>=0.6.0 cuda-python>=13.0.0
```

## How to run

Basic usage:
```bash
# Pre-steps: Set library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Run with default parameters (1M elements)
python simpleZeroCopy.py
```

With custom parameters:
```bash
# Use 2M elements
python simpleZeroCopy.py --num_elements 2097152

# Show help
python simpleZeroCopy.py --help
```

### Command line arguments

- `--num_elements`: Number of elements in vectors (default: 1048576)
  - Each vector uses `num_elements * 4 bytes` (float32)
  - Default: ~4 MB per vector, ~12 MB total

## Expected Output

Device name and compute capability **depend on your system**; the rest of the log should match this shape when validation passes.

```
======================================================================
simpleZeroCopy - CUDA Python Sample
======================================================================

Device Information:
  Name: <your GPU>
  Compute Capability: <major>.<minor>

> Memory: mapped pinned host (cudaHostAlloc + cudaHostGetDevicePointer)

Compiling CUDA kernel...
  Kernel compiled successfully

Allocating memory:
  Vector size: 1,048,576 elements
  Memory per vector: 4.00 MB
  Total memory: 12.00 MB

> Allocating mapped pinned host memory...
  Mapped host memory allocated successfully

> Initializing vectors on host...
> Computing reference result on CPU...

> Launching vectorAddGPU kernel...
  Note: GPU accesses host memory directly (zero-copy)
  Kernel execution complete

> Checking results from vectorAddGPU()...
  Comparing 1,048,576 elements...
  Relative error: 0.000000e+00
  Validation PASSED

======================================================================
simpleZeroCopy completed successfully!
======================================================================
```

## Files

- `simpleZeroCopy.py` – Main Python implementation
- `README.md` – This file
- `requirements.txt` – Python package dependencies
