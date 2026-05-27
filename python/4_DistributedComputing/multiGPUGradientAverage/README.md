# Sample: multiGPUGradientAverage (Python)

## Description

This sample demonstrates gradient averaging across multiple GPUs using MPI and cuda.core. Each GPU computes local gradients, which are synchronized (averaged) across all GPUs using MPI Allreduce with host-staging (GPU → CPU → MPI → CPU → GPU) for maximum compatibility.

## What you will learn

- How to initialize MPI for multi-process GPU communication
- How to map MPI ranks to CUDA devices consistently
- How to integrate cuda.core streams with CuPy using `ExternalStream`
- How to compile and launch custom CUDA kernels using cuda.core
- How to use cuda.core Event for GPU timing measurements
- How to use MPI Allreduce with host-staging for universal compatibility

## Prerequisites

- Python 3.10+
- CUDA Toolkit 13.0+
- Standard MPI implementation (OpenMPI, MPICH, or Intel MPI)
- Multiple NVIDIA GPUs (tested with 2+ GPUs)

## Installation

```bash
pip install mpi4py cupy-cuda13x cuda-python cuda-core
```

## Running

**IMPORTANT:** This sample **MUST** be run with `mpirun` with at least 2 processes.

```bash
# Single node (2 GPUs)
mpirun -np 2 python multiGPUGradientAverage.py --size 10000

# Single node (4 GPUs)
mpirun -np 4 python multiGPUGradientAverage.py --size 10000

# With specific GPUs
CUDA_VISIBLE_DEVICES=0,2 mpirun -np 2 python multiGPUGradientAverage.py
```

## Sample Output

```
[Rank 0] World size = 4

======================================================================
Multi-GPU Gradient Average Demo
======================================================================
Number of MPI ranks (GPUs): 4
Gradient vector length per GPU: 10000
Device: NVIDIA GeForce RTX 4090
Computation: gradients computed on GPU via cuda.core.
Communication: gradients averaged via MPI_Allreduce on host (CPU) buffers.
======================================================================

Sample averaged gradient values (rank 0):
  avg_grad[0] = 1.500000
  avg_grad[5000] = 6.500000
  avg_grad[9999] = 11.499000

Expected values:
  expected[0] = 1.500000
  expected[5000] = 6.500000
  expected[9999] = 11.499000

Verifying gradient averaging correctness...
[PASS] Gradient averaging is correct.
[PASS] Gradient averaging is correct on all ranks.

Performance:
  Kernel time (GPU only): 0.123 ms
  MPI communication time (host-staging, end-to-end): 0.456 ms
  Total time: 0.579 ms

======================================================================
Demo complete.
======================================================================
```

## Key Technical Details

The sample uses cuda.core streams and makes CuPy use them via `ExternalStream`:

```python
stream = device.create_stream()
cp.cuda.ExternalStream(int(stream.handle)).use()
```

GPU timing is measured using cuda.core Event:

```python
from cuda.core import EventOptions
timing_options = EventOptions(enable_timing=True)
start_event = stream.record(options=timing_options)
# ... GPU work ...
end_event = stream.record(options=timing_options)
end_event.sync()
kernel_time = end_event - start_event  # Returns milliseconds
```

The host-staging pattern transfers data GPU → CPU → MPI → CPU → GPU for universal MPI compatibility without requiring CUDA-aware MPI.

## Troubleshooting

**Error: "This sample requires at least 2 MPI processes!"**

Solution: Run with `mpirun -np 2 python multiGPUGradientAverage.py`
