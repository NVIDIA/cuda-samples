# Sample: Streaming Copy + Compute Overlap (Python)

## Description

Demonstrate how to overlap memory transfers (H2D/D2H) with kernel computation using CUDA streams. This technique hides transfer latency and improves GPU utilization.

## What You'll Learn

- Using `PinnedMemoryResource` for async-capable host memory
- Using `DeviceMemoryResource` for GPU memory allocation
- Creating multiple streams with `Device.create_stream()`
- Async memory copies with `Buffer.copy_to()`
- Overlapping H2D transfers, kernel execution, and D2H transfers

## Key Concept

**Without overlap (sequential):**
```
[====H2D====][====Compute====][====D2H====]
```

**With overlap (multiple streams):**
```
Stream 0: [H2D][Compute][D2H]
Stream 1:      [H2D][Compute][D2H]
Stream 2:           [H2D][Compute][D2H]
```

## Key APIs (all from `cuda.core`)

- `Device` - Device management
- `Device.create_stream()` - Create CUDA streams
- `Stream.sync()` - Synchronize stream
- `PinnedMemoryResource` - Pinned host memory (required for async transfers)
- `DeviceMemoryResource` - GPU device memory
- `Buffer.copy_to(dst, stream=stream)` - Async memory copy
- `Program`, `LaunchConfig`, `launch` - Kernel compilation and execution

### From `numpy`:

- `np.from_dlpack()` - Zero-copy view of pinned memory buffers

## Requirements

- CUDA Toolkit 13.0+
- Python 3.10+
- `cuda-python`, `cuda-core`, `numpy`

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python streamingCopyComputeOverlap.py
```

## Expected Output

```
============================================================
Streaming Copy + Compute Overlap
Using pure cuda.core APIs
============================================================

Device: NVIDIA GeForce RTX XXXX
Kernel compiled ✓

Problem size: 16,000,000 elements (61 MB)

--- Sequential (no overlap) ---
Timeline: [H2D][Compute][D2H]
Time: X.XX ms (±X.XX)

--- Streamed (with overlap) ---
Stream 0: [H2D][Compute][D2H]
Stream 1:      [H2D][Compute][D2H]
Stream 2:           [H2D][Compute][D2H]
...
2 streams: X.XX ms (±X.XX) - speedup: X.XXx
4 streams: X.XX ms (±X.XX) - speedup: X.XXx
8 streams: X.XX ms (±X.XX) - speedup: X.XXx

============================================================
Key: Pinned memory + multiple streams = overlap transfers with compute

Note: Speedup depends on hardware characteristics. This technique
benefits most when transfer time is significant relative to compute.
============================================================
```

## See Also

- [cuda.core Documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [CUDA Streams Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#overlapping-data-transfers)
