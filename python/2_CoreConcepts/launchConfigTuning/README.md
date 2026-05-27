# Sample: Launch Configuration Tuning (Python)

## Description

Benchmark different CUDA kernel launch configurations to find the optimal block-size setting using `cuda.core` APIs. This sample demonstrates **performance tuning** by measuring execution time across various thread block sizes.

## What You'll Learn

- Compiling CUDA kernels at runtime with `cuda.core.Program`
- Launching kernels with different `LaunchConfig` settings
- Benchmarking kernel performance with precise timing
- Understanding how thread block size affects performance
- Tuning for memory-bound vs compute-bound kernels

## Key Concepts

### Launch Configuration with cuda.core

```python
# Configure kernel launch with specific thread block size
config = LaunchConfig(
    grid=(grid_size,),
    block=(block_size,),
    shmem_size=shared_memory_bytes
)

# Launch kernel
launch(stream, config, kernel, *args)
stream.sync()
```

### Thread Block Sizing

Thread block size significantly impacts performance due to:

| Factor | Impact |
|--------|--------|
| **Occupancy** | More active warps can hide memory latency |
| **Registers** | More threads/block = fewer registers/thread |
| **Shared Memory** | Divided among blocks on each SM |
| **Warp Efficiency** | Block size should be multiple of 32 |

### Benchmarking Approach

```python
# Use CUDA events for accurate GPU timing (not CPU wall-clock)
start_event = device.create_event(options=EventOptions(enable_timing=True))
end_event = device.create_event(options=EventOptions(enable_timing=True))

stream.record(start_event)
for _ in range(n_iterations):
    launch(stream, config, kernel, *args)
stream.record(end_event)
end_event.sync()
elapsed_ms = (end_event - start_event) / n_iterations
```

## Key APIs

### From `cuda.core`:

- `Device` - CUDA device management
- `Program` - Runtime kernel compilation (NVRTC)
- `ProgramOptions` - Compilation options (architecture target)
- `LaunchConfig` - Kernel launch configuration (grid/block dimensions)
- `launch` - Execute compiled kernel (accepts Buffer objects directly)
- `EventOptions` - GPU timing with CUDA events
- `ManagedMemoryResource` - Device-preferred unified memory
- `ManagedMemoryResourceOptions` - Set preferred_location for representative benchmarks

### From `numpy`:

- `np.from_dlpack()` - Zero-copy view of GPU buffers via DLPack

### Benchmarked Kernels:

- **vector_add** - Simple memory-bound kernel (C = A + B) - low sensitivity to block size
- **reduce_sum** - Shared memory reduction - high sensitivity to block size

## Requirements

### Hardware:

- NVIDIA GPU with CUDA support
- Minimum GPU memory: 512 MB

### Software:

- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- See `requirements.txt` for Python packages

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python launchConfigTuning.py
```

## Expected Output

```
============================================================
Launch Configuration Tuning (cuda.core)
Finding the Best Block Size for Your Kernel
============================================================

Device: <Your GPU Name>
Compute Capability: X.X

Compiling CUDA kernels with cuda.core.Program...
  Target architecture: sm_XX
  ✓ vector_add kernel compiled
  ✓ reduce_sum kernel compiled

============================================================
VECTOR ADDITION - Launch Configuration Tuning
============================================================

Problem size: 10,000,000 elements
Kernel: vector_add (C = A + B)

Testing thread configurations: [32, 64, 128, 256, 512, 1024]
------------------------------------------------------------
Block Size:   32 | Blocks: 312500 | Time: X.XXXX ± X.XXXX ms
Block Size:   64 | Blocks: 156250 | Time: X.XXXX ± X.XXXX ms
...
------------------------------------------------------------

✓ OPTIMAL: block_size=XXX (X.XXXX ms)
✗ WORST:   block_size=XXX (X.XXXX ms)
  Speedup: X.XXx

✓ Results verified correct!

...

============================================================
SAMPLE COMPLETE
============================================================

Key Takeaway: The optimal thread configuration depends on your
specific kernel characteristics. Always benchmark to find the best!
```

## Tuning Guidelines

### Start Here
- **128-256 threads/block** is a good starting point for most kernels
- Always use **multiples of 32** (warp size)

### Memory-Bound Kernels
- Less sensitive to thread configuration
- Focus on memory access patterns
- Higher thread counts help hide latency

### Compute-Bound Kernels
- More sensitive to thread configuration
- Watch for register pressure at high thread counts
- Profile with Nsight Compute

### Reduction Kernels
- Block size affects shared memory usage
- Power-of-2 sizes simplify reduction logic
- Often 256-512 threads works well

## Files

- `launchConfigTuning.py` - Python implementation using cuda.core
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [cuda.core Documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [cuda.core.LaunchConfig](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.LaunchConfig.html)
- [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/)
- [CUDA Best Practices Guide - Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#execution-configuration-optimizations)
- [Nsight Compute Profiler](https://developer.nvidia.com/nsight-compute)
