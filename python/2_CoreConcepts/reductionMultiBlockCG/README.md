# Sample: Single-Pass Multi-Block Reduction with Cooperative Groups (Python)

## Description

Single-kernel, two-stage reduction using **Cooperative Groups** and `grid.sync()` so all blocks synchronize inside one launch—no second kernel or CPU stage for the reduction tree.

**Stack:** `cuda-core` (device, compile, cooperative `launch()`, stream, **CUDA events** for GPU timing). **CuPy** for H↔D copies on the same stream (`Stream.from_external(cuda.core_stream)`, `ndarray.data.ptr` to `launch()`). **`try`/`finally`** closes the stream if cooperative launch fails. Requires **compute capability > 6.0** (Pascal+).

## What you will learn

- `cooperative_groups::grid_group` and `grid.sync()` across the grid
- Cooperative `LaunchConfig(..., cooperative_launch=True)` and sizing blocks for residency
- Timing the GPU path with `EventOptions` / `stream.record()` / event elapsed time

## Key libraries

| Library | Role |
|---------|------|
| `cuda-core` | Device, stream, events, `Program` / `ProgramOptions`, cooperative `launch()` |
| `cupy` | `cp.empty`, `cp.asarray`, `cp.asnumpy`, `Stream.from_external` |
| `numpy` | Host data, reference sum, `default_rng` |

## Requirements

- NVIDIA GPU, **Pascal or newer**; **CUDA Toolkit 13+**; **Python 3.10+**
- NVRTC must see **`cooperative_groups.h`** and **CCCL** headers (`cuda/std/*`)

```bash
pip install -r requirements.txt
```

Pick a CuPy wheel that matches your CUDA major version (e.g. `cupy-cuda13x` in `requirements.txt`).

## How to run

**`--cuda-include-dir` is required** (colon-separated list). Typical desktop layout:

```bash
python reductionMultiBlockCG.py \
  --cuda-include-dir /usr/local/cuda/include/cccl:/usr/local/cuda/include
```

**Jetson / split include trees:** pass every directory NVRTC needs in one `--cuda-include-dir` argument, e.g.
`/usr/local/cuda/include/cccl:/usr/local/cuda/targets/sbsa-linux/include` (adjust paths to your image). If headers are scattered, you can instead merge them into one tree with symlinks and point `--cuda-include-dir` at that folder.

Defaults: **2²⁵** elements, threads = device max (capped at 1024), auto `--maxblocks`, **100** iterations. Other flags: `--n`, `--threads`, `--maxblocks`, `--iterations`. See **`python reductionMultiBlockCG.py --help`**.

## Output

```
======================================================================
Single-Pass Multi-Block Reduction with Cooperative Groups
======================================================================

Demonstrates: Multi-stage reduction in a single kernel using grid.sync()

Device Information:
  Name: NVIDIA Thor
  Compute Capability: sm_11.0

Reduction Configuration:
  Number of elements: 33,554,432
  Data size: 128.00 MB

Compiling CUDA kernel...
  Kernel compiled successfully

Launch Configuration:
  Threads per block: 1024
  Number of blocks: 20
  Total threads: 20,480
  Shared memory per block: 4096 bytes
  Launch mode: Cooperative (grid-wide sync enabled)

> Generating random input data...
> Computing reference result on CPU...
  CPU time: 0.008903 seconds

> Warming up GPU...
  Warm-up successful

> Running benchmark (100 iterations)...

> Performance Results:
  Average GPU time: 0.977166 ms
  Throughput: 137.35 GB/s
  Speedup vs CPU: 9.11x

> Validating results...
Test PASSED

======================================================================
Summary
======================================================================

Single-kernel two-stage reduction:
  Stage 1: 20 blocks → 20 partial sums
  grid.sync() ← All blocks synchronize (KEY innovation)
  Stage 2: Block 0 → 1 final result
  Total: 1 kernel launch, 137.35 GB/s

Comparison:
  • Traditional: 2 kernel launches or kernel + CPU
  • This sample: 1 kernel with grid.sync() between stages
  • Benefit: Eliminates ~5-20μs launch overhead per stage

======================================================================
Single-Pass Multi-Block Reduction completed successfully!
======================================================================
```

## Troubleshooting (short)

- **Cooperative launch not supported / fails:** need sm_60+; reduce `--maxblocks` or `--threads` so all blocks can be resident.
- **Compile errors missing headers:** extend `--cuda-include-dir` with the path that contains CCCL / cooperative groups (see Jetson note above).
- **Low throughput:** often block count vs occupancy; try defaults first, then tune `--threads` / `--maxblocks`.

## Related samples

**blockArraySum** (atomics + grid-stride) → **reduction** (two-stage shared memory) → **this sample** (single kernel + `grid.sync()`).

## Further reading

- [CUDA Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
- [Reduction whitepaper (PDF)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

## Files

`reductionMultiBlockCG.py` · `requirements.txt` · `README.md`
