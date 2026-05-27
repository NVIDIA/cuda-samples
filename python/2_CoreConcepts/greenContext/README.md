# greenContext (Python)

## Description

This sample demonstrates how to use **green contexts** with
`cuda.core` to statically partition a GPU's streaming multiprocessors
(SMs) so that independent kernels can run on dedicated subsets of the
device.

This examples takes A long-running kernel that fills the GPU's SMs,
and a short but latency-sensitive "critical" kernel is launched shortly after.
Without green contexts, the critical kernel must wait for SMs to
free up. With green contexts, the GPU's SMs are partitioned so the
critical kernel has its own dedicated SMs and can start immediately.

Three timed scenarios are compared:

1. **Reference**: the critical kernel alone on the primary context,
   with no competing work. Establishes the pure compute time of the
   critical kernel when every SM on the device is available to it.
2. **Baseline**: both kernels run on the device's primary context,
   on two non-blocking streams that contend for all SMs.
3. **Green contexts**: the SMs are split into two disjoint groups
   (e.g. 112 + 16). Each kernel runs on a stream that belongs to its
   own green context, so the critical kernel never waits for SMs
   held by the long-running kernel.

The headline metric is the total wall time of the critical kernel
from launch to completion. In the baseline it is dominated by time
spent waiting behind the long-running kernel. With green contexts it
reflects the kernel's own compute time on its (smaller) SM
partition. The reference row lets you separate those two effects:

- `baseline - reference` is roughly the time the critical kernel
  spent waiting for SMs in the baseline run (the cost that green
  contexts eliminate).
- `green / reference` is the compute slowdown caused by running on
  a smaller SM partition (the cost that green contexts introduce).

## What You'll Learn

- Querying a device's SM resources via `Device.resources.sm` and
  reading `sm_count`, `min_partition_size`, `coscheduled_alignment`
- Splitting an `SMResource` into disjoint partitions with
  `sm.split(SMResourceOptions(count=(A, B)))`
- Creating a green context from an SM partition via
  `Device.create_context(ContextOptions(resources=[group]))`
- Creating a non-blocking stream on a green context with
  `ctx.create_stream()`
- Using CUDA events with timing enabled to measure kernel wall time
  across streams
- Cleaning up green contexts safely with `ctx.close()`

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - device management, SM partitioning, green contexts, compilation, and launching
- `numpy` - scalar kernel arguments

## Key APIs

### From `cuda.core`

- `Device.resources.sm` - the device's SM-type device resource
- `SMResource.split(SMResourceOptions(count=(A, B)))` - partition SMs
  into disjoint groups (plus an optional remainder)
- `Device.create_context(ContextOptions(resources=[sm_group]))` -
  create a green context provisioned with a specific SM partition
- `Context.is_green` / `Context.resources` - introspect a green
  context
- `Context.create_stream()` - create a non-blocking stream that is
  tied to the green context's SM partition
- `Context.close()` - destroy a green context (must not be the
  thread's current context when closed)
- `Device.create_event(EventOptions(timing_enabled=True))` /
  `Stream.record(event)` / `event2 - event1` - measure elapsed time
  in milliseconds between two events on the device
- `Program(..., ProgramOptions(std="c++17", arch=f"sm_{device.arch}"))`
  / `program.compile("cubin", name_expressions=(...))` - compile the
  delay and critical kernels in one TU
- `launch(stream, LaunchConfig(grid=..., block=...), kernel, ...)` -
  submit a kernel on a specific stream

## Requirements

### Hardware

- Any NVIDIA GPU supported by green contexts.
- Green-context SM partitioning is designed for larger server GPUs
  (H100, H200, B200, ...) but works on any supported GPU as long as
  the SM count is large enough to split meaningfully.

### Software

- NVIDIA driver >= 12.4
- CUDA Toolkit 13.0 or newer.
- Python 3.10 or newer.
- `cuda-core` (`>=1.0.0`)

## Installation

Install the required packages from `requirements.txt`:

```bash
cd /path/to/cuda-samples/python/2_CoreConcepts/greenContext
pip install -r requirements.txt
```

## How to Run

### Basic usage

The auto-default split reserves a small partition (~16 SMs) for the
critical kernel and gives the rest to the long-running kernel. The
exact sizes are chosen by probing the driver with a dry-run `sm.split`,
escalating the alignment granularity in powers of two until the driver
accepts the pair. This handles architectures where the driver enforces
stricter alignment (e.g. TPC/GPC-pair alignment on Blackwell) than the
reported `min_partition_size`. When that happens the sample prints a
`Note:` line with the granularity it landed on.

```bash
cd cuda-samples/python/2_CoreConcepts/greenContext
python greenContext.py
```

### Match the CUDA programming guide example (112 + 16)

```bash
python greenContext.py --split 112,16
```

### Tune the workload

```bash
# Longer long-running kernel, larger host launch gap
python greenContext.py --delay-us 3000 --launch-gap-ms 2.0

# Smaller/lighter critical kernel so its own compute time is negligible
python greenContext.py --critical-n 65536 --critical-iters 128

# Symmetric split: maximum SMs for the critical kernel, long kernel is
# roughly 2x slower but the critical kernel runs close to its reference time.
python greenContext.py --split 64,64

# Use a specific GPU
python greenContext.py --device 1
```

### All options

```
--device           CUDA device ID (default: 0)
--split            SM split as 'LONG,CRITICAL', e.g. '112,16'.
                   Each side must be a multiple of the device's
                   min_partition_size, and the driver may enforce additional
                   architecture-specific alignment (e.g. TPC/partition-grid
                   alignment on Blackwell). Omit --split to auto-select a
                   driver-accepted split.
--delay-us         Per-block busy-wait of the delay kernel, in us (default: 2000)
--delay-waves      Number of waves of the delay kernel on the long
                   partition. Drives the default --delay-blocks (default: 16)
--delay-blocks     Number of blocks for the delay kernel. Overrides
                   --delay-waves if set.
                   (default: --delay-waves * device SM count)
--critical-n       Work size of the critical kernel (default: 4194304)
--critical-iters   Inner math-loop iterations inside the critical kernel.
                   Higher values make the critical kernel's own compute
                   time more substantial relative to its wait time
                   (default: 1024)
--launch-gap-ms    Host delay between launching the long and critical
                   kernels (default: 1.0 ms)
```

## Expected Output

The output depends on the GPU and the number of SMs.
On an RTX 4090 (128 SMs) with the default auto split:

```
[Green Context Sample using CUDA Core API]
Device: NVIDIA GeForce RTX 4090
Compute Capability: sm_89
Total SMs:                 128
Min. SM partition size:    2
SM co-scheduled alignment: 2
SM split (long/critical):  112 / 16
Workload parameters:
  delay kernel: 2048 blocks, 2000 us/block (~32.0 ms on 128 SMs)
  critical kernel: 4194304 elements, 1024 inner iterations
  host launch gap: 1.0 ms

Compiling kernels ...
Running reference scenario (critical kernel alone) ...
Running baseline scenario (primary context) ...
Running green context scenario ...

scenario                             SMs (long/crit)     long (ms)   crit total (ms)   crit offset (ms)
-------------------------------------------------------------------------------------------------------
crit alone (primary ctx)                       -/128             -             0.425                  -
baseline (primary ctx)                       128/128        32.034            30.024              1.090
green ctx (112+16 SMs)                        112/16        38.017             2.696              1.075

long (ms)        : wall time of the delay kernel
crit total (ms)  : launch-to-complete wall time of the critical kernel
crit offset (ms) : when the critical stream started, relative to the long stream start

Critical-kernel latency speedup (baseline vs green ctx): 11.14x
Green-ctx compute cost vs unconstrained (crit alone):    6.34x
Baseline time spent waiting for SMs (not computing):     ~29.60 ms

Done
```

**What to look for:**

- The critical kernel alone (reference row) takes only a fraction of
  a millisecond; almost all of the baseline's `crit total` is time
  spent queued waiting for SMs, not compute.
- The **critical kernel's wall time drops sharply** in the
  green-context scenario (from ~30 ms to a few ms in the example
  above) because it no longer waits for SMs held by the long-running
  kernel.
- The **long-running kernel's duration may increase** proportional
  to the reduction in SMs available to it (128 -> 112 SMs ~= 14%
  slower; 128 -> 64 SMs ~= 2x slower). This is an expected tradeoff:
  you reserve SMs for a critical kernel by taking them away from the
  background workload.
- The **compute cost** ratio (`green / reference`) shows how close
  the critical kernel is to ideal linear scaling with its SM count.
  A 112/16 split gives the critical kernel only 12.5% of the SMs and
  costs it roughly 6-7x its reference time; a 64/64 split gives it
  half the SMs and costs roughly 1.5-2x.
- The `crit offset` column is approximately `--launch-gap-ms` in
  both full scenarios; it confirms the host launched the critical
  kernel the same amount of time after the long kernel in both runs.

Exact timings vary with GPU model, driver version, clock state, and
other concurrent GPU work.

## Files

- `greenContext.py` - Python implementation using `cuda.core` green-context APIs
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [Green Contexts in the CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#green-contexts)
- [`cuda.core` green-context test suite](https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/tests/test_green_context.py) - the authoritative API reference
