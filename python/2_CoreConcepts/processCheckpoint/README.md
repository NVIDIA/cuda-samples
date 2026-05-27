# processCheckpoint (Python)

## Description

This sample demonstrates how to use the **CUDA process checkpoint API**
via `cuda.core.checkpoint.Process` to suspend, capture, and restore the
CUDA state of a running Linux process.

CUDA process checkpointing is the driver-level primitive that powers
CRIU + `cuda-checkpoint` integration.

The sample:

1. Allocates a GPU buffer and fills it with a deterministic pattern
   via a small kernel.
2. Reads the buffer back to host and computes a SHA-256 hash.
3. Runs the full checkpoint lifecycle on its own process:
   `lock → checkpoint → restore → unlock`.
4. Reads the buffer back again and verifies that the hash is
   unchanged, proving that GPU memory contents survived the round
   trip.

The sample prints the CUDA process state after each step so the
full state machine is visible:

```
         lock()          checkpoint()              restore()         unlock()
running ---------> locked ------------> checkpointed -----------> locked ---------> running
```

## What You'll Learn

- Creating a `cuda.core.checkpoint.Process` for the current process
  by PID and observing its `.state` transitions.
- Running the full `lock → checkpoint → restore → unlock` cycle with
  a lock timeout.
- The fact that `restore()` leaves the process in the `locked` state;
  you must still call `unlock()` to return to `running`.
- Verifying that GPU memory is preserved across the checkpoint
  round-trip by comparing SHA-256 hashes of the buffer before and
  after.
- The rough cost of each step (checkpoint and restore dominate and
  scale with the device-memory footprint being captured).

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/)
  - device management, memory allocation, kernel compilation and
    launch, and the `checkpoint.Process` wrapper.
- [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/)
  - used directly for a pageable `cuMemcpyDtoH`.

## Key APIs

### From `cuda.core.checkpoint`

- `checkpoint.Process(pid)` - create a handle to a CUDA process by
  PID. Accepts `os.getpid()` for the self-checkpoint case shown
  here.
- `Process.state` - one of `"running"`, `"locked"`, `"checkpointed"`,
  or `"failed"`.
- `Process.lock(timeout_ms=…)` - block further CUDA API calls on the
  process; completes already-submitted work. Always pass a non-zero
  timeout to avoid deadlocks.
- `Process.checkpoint()` - copy device memory to host-side driver
  allocations and release GPU resources. Process state becomes
  `checkpointed`.
- `Process.restore(gpu_mapping=None)` - re-acquire GPU resources and
  copy memory back to device. Leaves the process in the `locked`
  state.
- `Process.unlock()` - return the process to `running`.
- `Process.restore_thread_id` - thread ID that `restore()` must be
  called from in the target process (not used in the self-checkpoint
  case here).

### From `cuda.core`

- `Device.set_current()` / `Device.memory_resource.allocate(...)` /
  `Stream`, `LaunchConfig`, `Program`, `launch` - standard device,
  compile, and launch primitives used to produce the buffer
  contents.

### From `cuda.bindings.driver`

- `cuMemcpyDtoH(host_ptr, device_handle, nbytes)` - synchronous D2H
  copy into a pageable host buffer.

## Requirements

### Hardware

- Any NVIDIA GPU supported by CUDA process checkpointing. CUDA
  checkpointing is currently limited to x86-64 Linux.

### Software

- Linux (the CUDA checkpoint API is Linux-only).
- NVIDIA driver with CUDA process checkpoint support.
- CUDA Toolkit 13.0 or newer.
- Python 3.10 or newer.
- `cuda-core >= 1.0.0`.

## Installation

Install the required packages from `requirements.txt`:

```bash
cd /path/to/cuda-samples/python/2_CoreConcepts/processCheckpoint
pip install -r requirements.txt
```

## How to Run

### Basic usage

```bash
python processCheckpoint.py
```

### Larger GPU footprint to see checkpoint time scale

```bash
python processCheckpoint.py --buffer-mib 512
```

### Use a specific GPU

```bash
python processCheckpoint.py --device 1
```

### All options

```
--device           CUDA device ID (default: 0)
--buffer-mib       GPU buffer size in MiB (default: 16)
--lock-timeout-ms  Timeout passed to Process.lock in ms (default: 5000)
```

## Expected Output

On an RTX 4090 with a 16 MiB buffer:

```
[Process Checkpoint Sample using CUDA Core API]
PID:                748330
Device:             NVIDIA GeForce RTX 4090
Compute Capability: sm_89
Buffer size:        16 MiB
Lock timeout:       5000 ms

Compiling kernel ...
Writing deterministic pattern to GPU buffer ...
Buffer hash (before): b045f7975dc23352

Running checkpoint lifecycle on self ...

step               duration (ms)       state after
--------------------------------------------------
initial                        -           running
lock                       0.578            locked
checkpoint               268.369      checkpointed
restore                  235.024            locked
unlock                     1.648           running
--------------------------------------------------
total                    505.618

Buffer hash (before): b045f7975dc23352
Buffer hash (after):  b045f7975dc23352

PASS: GPU buffer contents survived checkpoint/restore.

Done
```

**What to look for:**

- The **four state transitions** are all observable: `running →
locked → checkpointed → locked → running`. Note that `restore()`
  leaves the process in `locked`, not `running`.
- The **checkpoint and restore steps dominate** the wall-clock time
  (hundreds of ms even for a small buffer) - they copy GPU memory to
  and from driver-managed host allocations. Increasing
  `--buffer-mib` visibly increases the checkpoint time.
- The `lock` and `unlock` steps are essentially free (sub-ms) - they
  just flip the process state.
- The SHA-256 **hashes before and after match**, proving the GPU
  memory contents survived the round trip.

Exact timings vary with GPU model, driver version, system load, and
the size of the device memory footprint being captured.

## Files

- `processCheckpoint.py` - Python implementation using `cuda.core.checkpoint`
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`NVIDIA/cuda-checkpoint`](https://github.com/NVIDIA/cuda-checkpoint)
  - the CUDA checkpoint/restore utility, the CRIU plugin, and C
    reference programs (`r570-features.c`, `r580-migration-api.c`).
- [Checkpointing CUDA Applications with CRIU](https://developer.nvidia.com/blog/checkpointing-cuda-applications-with-criu/)
  - NVIDIA technical blog post on the broader CRIU workflow.
