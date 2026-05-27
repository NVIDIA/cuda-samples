# simplePrint - Printing from CUDA Kernels

## Description

This sample demonstrates how to use `printf()` inside CUDA kernels using **two different approaches**:

1. **CUDA C++ kernels** compiled with `cuda.core.Program` - Full C++ features and control
2. **Numba CUDA kernels** - Pythonic kernel authoring using `numba.cuda.grid()` for modern indexing

The sample shows basic device management, kernel compilation with inline CUDA C++ code, and multi-dimensional kernel launches (2D grid × 3D blocks) using modern CUDA Python. The Numba example demonstrates the recommended `numba.cuda.grid()` indexing style while also showing how it relates to classic CUDA C++ block/thread IDs. Both approaches use `cuda.core` APIs for stream management and synchronization, demonstrating interoperability.

This is the Python equivalent of the C++ `simplePrintf` sample, enhanced with Numba CUDA examples.

## Key Concepts

CUDA Python (cuda.core), Numba CUDA, Kernel Compilation, Printf in Kernels, Multi-dimensional Launch, Pythonic GPU Programming, Modern Thread Indexing (grid()), Stream-based Execution, cuda.core/Numba Interoperability

## CUDA APIs involved

### [cuda.core (cuda-python)](https://nvidia.github.io/cuda-python/)

- `Device()` - Device management
- `Device.create_stream()` - Create CUDA streams
- `Stream.sync()` - Synchronize stream execution
- `Program()` - Compile CUDA C++ kernels
- `LaunchConfig()` - Configure kernel launch
- `launch()` - Execute kernels on streams

### [Numba CUDA](https://nvidia.github.io/numba-cuda/)

- `@cuda.jit` - JIT compile Python functions to CUDA kernels
- `cuda.grid()` - Get global thread position (recommended modern approach)
- `cuda.blockIdx`, `cuda.threadIdx` - Thread/block indices (classic style)
- `cuda.gridDim`, `cuda.blockDim` - Grid/block dimensions
- **Note:** Uses `cuda.core` APIs for stream management (interoperability)

### CUDA Kernel Functions

- `printf()` - Print from device code (C++)
- `print()` - Print from device code (Numba, limited formatting)
- `blockIdx`, `threadIdx` - Thread/block indices
- `gridDim`, `blockDim` - Grid/block dimensions

### What You Learn

- Device initialization with `cuda.core.Device`
- Compiling CUDA C++ kernels with `Program` and `ProgramOptions`
- Writing Pythonic CUDA kernels with Numba's `@cuda.jit` decorator
- Using `numba.cuda.grid()` for modern thread indexing (recommended approach)
- Understanding the relationship between global coordinates and classic block/thread IDs
- **Interoperability**: Using `cuda.core` streams with Numba CUDA kernels
- Comparing CUDA C++ vs Pythonic kernel authoring approaches
- Multi-dimensional kernel launches (2D grid, 3D blocks)
- Using streams for kernel execution and synchronization
- Using `printf()` and `print()` in GPU kernels for debugging
- Understanding print limitations in Numba CUDA (no f-strings)
- Proper error handling and resource management

## Requirements

### Hardware:

- NVIDIA GPU with Compute Capability 7.0 or higher
- Minimum GPU memory: 512 MB

### Software:

- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- `cuda-python` package (13.0+)
- `cuda-core` package (>=0.6.0)
- `numba-cuda` package (0.24.0+, for Pythonic kernel authoring)

Download and install:
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [cuda-python package](https://nvidia.github.io/cuda-python/): `pip install cuda-python`
- [numba-cuda](https://nvidia.github.io/numba-cuda/): `pip install numba-cuda`

## Build and Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the sample
python simplePrint.py
```

## Expected Output

```
Simple Print - Printing from CUDA Kernels
Demonstrating both CUDA C++ and Numba CUDA approaches

Device: <Your GPU Name>
Compute Capability: sm_<XX>

======================================================================
METHOD 1: CUDA C++ Kernel (via cuda.core.Program)
======================================================================
Advantage: Full C++ features, better for complex kernels

Compiling CUDA C++ kernel...
Kernel compiled successfully.

Kernel configuration:
  Grid:  (2, 2)
  Block: (2, 2, 2)
  Total threads: 32

Launching kernel with value=10. Output:

[0, 0]:		Value is: 10
[0, 1]:		Value is: 10
[0, 2]:		Value is: 10
[0, 3]:		Value is: 10
[0, 4]:		Value is: 10
[0, 5]:		Value is: 10
[0, 6]:		Value is: 10
[0, 7]:		Value is: 10
[1, 0]:		Value is: 10
...
[3, 7]:		Value is: 10

CUDA C++ kernel execution complete.


======================================================================
METHOD 2: Numba CUDA Kernel (Pythonic / modern indexing)
======================================================================
Advantage: Uses numba.cuda.grid(3) for global indexing,
           while still showing classic CUDA C++ IDs for reference.
           Uses cuda.core for stream management (interoperability).

Kernel configuration:
  Grid:  (2, 2)
  Block: (2, 2, 2)
  Total threads: 32

Launching Numba CUDA kernel (grid(3) + classic IDs) with value=10:
Uses numba.cuda.grid(3) to get global (x, y, z),
and prints the corresponding blockId/threadId like the C++ sample.
Stream managed by cuda.core for consistency with C++ example.

global[ 0 , 0 , 0 ]  -> [ 0 ,  0 ]: Value is: 10
global[ 1 , 0 , 0 ]  -> [ 0 ,  1 ]: Value is: 10
global[ 0 , 1 , 0 ]  -> [ 0 ,  2 ]: Value is: 10
...
global[ 3 , 3 , 1 ]  -> [ 3 ,  7 ]: Value is: 10

Numba CUDA kernel execution complete.

======================================================================
Done! Both kernel approaches demonstrated successfully.
======================================================================
```

## Understanding the Output

- **Grid**: 2×2 = 4 blocks (labeled 0-3)
- **Block**: 2×2×2 = 8 threads per block (labeled 0-7)
- **Total**: 32 threads, each printing its position and value

### CUDA C++ Kernel:
Each thread calculates:
- Block ID (linear): `blockIdx.y * gridDim.x + blockIdx.x`
- Thread ID (linear): `threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x`

### Numba CUDA Kernel:
Each thread shows:
- **Global position** using `numba.cuda.grid(3)` → `(x, y, z)` coordinates across entire grid
- **Classic IDs** (block ID, thread ID) calculated the same way as C++ for comparison
- This demonstrates how modern indexing relates to traditional CUDA C++ style

### Comparing the Two Approaches

**CUDA C++ Kernel (Method 1):**
- Uses C++ syntax and `printf()` with full formatting control
- Requires compilation via `cuda.core.Program`
- Best for complex kernels needing C++ features (templates, libraries, etc.)
- Uses classic block/thread ID indexing
- Output: `[0, 0]:		Value is: 10` (clean formatting)

**Numba CUDA Kernel (Method 2):**
- Uses Python syntax with `@cuda.jit` decorator
- JIT compiled automatically when called
- Best for prototyping and simpler kernels
- **Modern indexing**: Uses `numba.cuda.grid(3)` to get global thread coordinates (recommended)
- Also shows classic block/thread IDs to help relate the two indexing models
- **Interoperability**: Uses `cuda.core` streams via `stream` for consistency
- Demonstrates that numba-cuda kernels can work seamlessly with cuda.core infrastructure
- Limited print formatting (no f-strings, basic `print()` only; adds spaces between arguments)
- Output: `global[ 0 , 0 , 0 ] -> [ 0 , 0 ]: Value is: 10` (shows both indexing styles; note extra spaces due to `print()` behavior)

## Experiments

Try modifying:

### For Both Approaches:
- **Grid size**: Change `grid=(4, 4)` for 16 blocks
- **Block size**: Change `block=(4, 4, 4)` for 64 threads per block
- **Conditional printing**: Print only from specific threads (e.g., `if threadId == 0:`)

### CUDA C++ Specific:
- **Format strings**: Experiment with different `printf()` formats
- **Kernel code**: Add complex C++ computations before printing
- **External libraries**: Include CUDA math libraries or device functions (e.g., `<cuda/std/cmath>`, `<cub/cub.cuh>`)

### Numba CUDA Specific:
- **Grid indexing**: Try `numba.cuda.grid(1)` or `numba.cuda.grid(2)` for different dimensions
- **Conditional printing**: Print only from threads where `x == 0` or `y == z`
- **Python operations**: Use NumPy-like operations in the kernel
- **Device math libraries**: Use [nvmath-python device APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/index.html) for optimized math operations (similar to CUDA math libraries in C++)
- **Shared memory**: Add `numba.cuda.shared.array()` for fast inter-thread communication
- **Atomic operations**: Try `numba.cuda.atomic.add()` for thread-safe updates
- **Print variations**: Experiment with what numba-cuda's `print()` can and cannot handle
- **Streams**: Create multiple `cuda.core` streams and launch numba-cuda kernels on them concurrently
- **Interoperability**: Mix numba-cuda kernels and CUDA C++ kernels on the same stream

## Notes

### General:
- Printing from GPU is relatively slow - use sparingly in production code
- Printf output is buffered and limited (~1MB buffer on most GPUs)

### CUDA C++ Kernels:
- Always call `stream.sync()` after kernel launch to flush printf output
- Full `printf()` format string support (%, flags, width, precision)

### Numba CUDA Kernels:
- **Recommended**: Use `numba.cuda.grid(ndim)` for thread indexing (modern, Pythonic)
  - `grid(1)` for 1D indexing, `grid(2)` for 2D, `grid(3)` for 3D
  - Returns global thread position across the entire grid
- **Interoperability**: Use `cuda.core` streams with Numba kernels via `stream`
  - Create streams: `stream = device.create_stream()`
  - Launch kernels: `kernel[grid, block, stream](args)`
  - Synchronize: `stream.sync()`
- Numba's `print()` has limited capabilities compared to Python's `print()`
- F-strings are NOT supported in Numba CUDA kernels
- Use comma-separated arguments: `print("Value:", x)` instead of f-strings
- **Note**: `print()` automatically adds spaces between comma-separated arguments (e.g., `print("[", x, "]")` outputs `[ 0 ]` not `[0]`)
- Always synchronize the stream to flush output

## Files

- `simplePrint.py` - Python implementation using cuda.core API
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

### CUDA Python (cuda.core):
- [cuda.core Documentation](https://nvidia.github.io/cuda-python/)
- [CUDA Python Examples](https://github.com/NVIDIA/cuda-python/tree/main/cuda_core/examples)

### Numba CUDA:
- [Numba CUDA Documentation](https://nvidia.github.io/numba-cuda/)
- [numba.cuda.grid() Reference](https://nvidia.github.io/numba-cuda/reference/kernel.html#numba.cuda.grid)
- [nvmath-python Device APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/index.html) - Optimized math operations for Numba CUDA kernels

### CUDA References:
- [CUDA C Programming Guide - Printf](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#formatted-output)
- [C++ simplePrintf Sample](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simplePrintf)
