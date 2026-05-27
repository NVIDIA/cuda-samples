# Sample: Fast Array Sum using Shared Memory (Python)

## Description

Two-stage parallel reduction: each GPU block sums its chunk in **shared memory** (tree reduction, two elements per thread), writes one partial sum per block; the host combines partial sums for the final result.

**Stack:** `cuda-core` for `Device`, stream, events, `Program` / `launch()`. **CuPy** allocates device memory and copies; `launch()` takes device pointers as `ndarray.data.ptr` (Python `int`). Copies run on the same CUDA stream as the kernel via `cp.cuda.Stream.from_external(stream)` (cuda.core `Stream` implements the CUDA stream protocol) and `with cp_stream:`.

## What you will learn

- Shared-memory block reduction and sequential-addressing tree reduction
- `LaunchConfig` with dynamic shared memory and `launch()` with pointer arguments
- Aligning CuPy transfers with a `cuda.core` stream (`Stream.from_external`)
- GPU timing with `EventOptions` / `device.create_event()`

## Key libraries

| Library    | Role |
|------------|------|
| `cuda-core`| Device, stream, events, compile, launch |
| `cupy`     | `cp.empty`, `cp.asarray`, `cp.asnumpy`, `Stream.from_external` |
| `numpy`    | Host data and CPU reference sum |

## Key APIs (quick reference)

- **cuda.core:** `Device`, `create_stream`, `Program` / `ProgramOptions`, `LaunchConfig`, `launch`, `EventOptions`, `create_event`
- **CuPy:** `cp.empty`, `cp.asarray`, `cp.cuda.Stream.from_external(stream)`, `with cp_stream:`, `cp.asnumpy`

## Requirements

- NVIDIA GPU, CUDA-capable driver; **CUDA Toolkit 13+** (for toolchain alignment with `cuda-core`)
- **Python 3.10+**

```bash
pip install -r requirements.txt
```

## How to run

```bash
python reduction.py
```

Defaults: 2²⁴ elements, 256 threads/block, `float`, 100 benchmark iterations.

**Change data type** (selects `blockReduceKernel_int` / `_float` / `_double`):

```bash
python reduction.py --type float    # default; 32-bit float
python reduction.py --type double   # 64-bit float
python reduction.py --type int      # 32-bit integer (exact equality check)
```

Combine with other flags as needed, e.g. `python reduction.py --type int --n 1048576`.

Other main flags: `--n`, `--threads`, `--iterations`. Full list: `python reduction.py --help`.

## Output

Example run (`python reduction.py`, defaults) on **Tesla T10**, compute capability **7.5**:

```
======================================================================
Fast Array Sum using Shared Memory - Two-Stage Reduction
======================================================================

Demonstrates: Efficient parallel reduction using shared memory

Device Information:
  Name: Tesla T10
  Compute Capability: sm_7.5

Configuration:
  Array size: 16,777,216 elements
  Data type: float
  Memory: 64.00 MB
  Threads per block: 256

Two-Stage Reduction Strategy:
  Stage 1: GPU block reduction
    - Number of blocks: 32768
    - Elements per block: 512
    - Output: 32768 partial sums
  Stage 2: CPU final reduction
    - Combine 32768 partial sums → 1 final result

Compiling CUDA kernel...
  Kernel 'blockReduceKernel_float' compiled successfully

> Generating random input data...
> Computing reference result on CPU...
  CPU time: 2.428208 seconds

> Warming up GPU...
  Warm-up completed

> Benchmarking Stage 1 (GPU block reduction)...
  Running 100 iterations...

> Running Stage 2 (CPU final reduction)...

======================================================================
Performance Results
======================================================================

Stage 1 (GPU block reduction):
  Average time: 0.338404 ms
  Throughput: 198.31 GB/s

Stage 2 (CPU final reduction):
  Time: 0.078073 ms
  (32768 partial sums)

Total time: 0.416477 ms
Speedup vs CPU: 5830.35x

> Validating results...
  GPU result: 2147639808.00000000
  CPU result: 2147639929.62027407
Test PASSED

======================================================================
Summary
======================================================================
Key optimizations:
  - Load 2 elements per thread: 8,388,608 global reads (50% savings)
  - Shared memory for reduction: ~10-20x faster than global memory
  - Parallel block outputs: 32768 independent writes
Result: 198.31 GB/s throughput
======================================================================
Two-Stage Reduction completed successfully!
======================================================================
```

## Files

`reduction.py` · `requirements.txt` · `README.md`
