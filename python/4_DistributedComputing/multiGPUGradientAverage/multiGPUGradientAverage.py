# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    distribution and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Multi-GPU Gradient Average using MPI and cuda.core (Host-staging Allreduce)

Question: How do I synchronize gradients across GPUs?

Answer:
Each GPU (each MPI rank) computes local gradients on device via CUDA.
Gradients are then averaged across ranks via an MPI Allreduce over host
(CPU) buffers, following the classic data-parallel training pattern.

This sample shows how to:
- Initialize MPI for multi-process GPU workloads
- Map MPI ranks to GPUs
- Use cuda.core for kernel compilation and execution
- Integrate cuda.core with CuPy using the stream protocol
- Perform gradient averaging with MPI Allreduce (using host staging)
- Use cuda.core Event for GPU timing measurements
- Verify correctness of distributed gradient synchronization

Key concepts: Allreduce, NCCL collectives (conceptually), distributed training

Note:
- All gradient computation and validation happen on GPUs.
- MPI Allreduce is executed on CPU (host) buffers via a simple
  GPU -> CPU -> MPI -> CPU -> GPU staging pattern so that the sample
  works on any MPI stack, without requiring CUDA-aware MPI.
- In production deep learning frameworks (e.g., PyTorch DDP), NCCL
  usually implements the GPU Allreduce directly; the communication
  pattern and semantics are the same as demonstrated here.
"""

import sys
from pathlib import Path

# Add parent directory to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))
from cuda_samples_utils import verify_array_result  # noqa: E402

try:
    import cupy as cp
    from cuda.core import (
        Device,
        EventOptions,
        LaunchConfig,
        Program,
        ProgramOptions,
        launch,
        system,
    )
    from mpi4py import MPI
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install: pip install mpi4py cupy-cuda12x cuda-python cuda-core")
    sys.exit(1)


# ============================================================================
# CUDA device selection and stream management
# ============================================================================


def init_device(rank: int):
    """
    Initialize CUDA device and stream for this MPI rank.

    For a simple single-node run, we map rank % num_gpus to a device id.
    This covers both the common case (world_size == num_gpus) and the case
    where multiple ranks share a GPU.

    Returns
    -------
    tuple[Device, Stream]
        CUDA device object and stream object.
    """
    num_gpus = system.get_num_devices()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices available")

    dev_id = rank % num_gpus  # simple mapping: rank -> GPU in round-robin

    try:
        device = Device(dev_id)
    except (RuntimeError, ValueError) as e:
        if rank == 0:
            print(f"Warning: Cannot assign GPU {dev_id}, using GPU 0. Error: {e}")
        device = Device(0)

    device.set_current()
    # Align CuPy with cuda.core's chosen device ID
    cp.cuda.Device(device.device_id).use()

    # Create cuda.core stream and make CuPy use it
    stream = device.create_stream()
    cp.cuda.ExternalStream(int(stream.handle)).use()

    return device, stream


# ============================================================================
# CUDA kernel definition and compilation
# ============================================================================

# Tiny CUDA kernel to initialize local "gradients"
# Uses grid-stride loop to handle arrays larger than grid size
INIT_KERNEL = r"""
extern "C" __global__
void init_grad_kernel(float* grad, int n, int rank)
{
    // Grid-stride loop: each thread processes multiple elements
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = tid; i < n; i += stride) {
        // Gradient value depends on MPI rank so we can verify reduction:
        // grad_i = rank + 0.001 * i
        grad[i] = rank + 0.001f * i;
    }
}
"""

_kernel_cache = {}


def get_init_kernel(device: Device):
    """Compile (or retrieve cached) init_grad_kernel for this device."""
    key = device.pci_bus_id
    if key not in _kernel_cache:
        opts = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
        prog = Program(INIT_KERNEL, code_type="c++", options=opts)
        mod = prog.compile("cubin")
        _kernel_cache[key] = mod.get_kernel("init_grad_kernel")
    return _kernel_cache[key]


# ============================================================================
# Local gradient computation on each GPU
# ============================================================================


def compute_local_gradients(
    num_elements: int, device: Device, stream: object, rank: int
) -> cp.ndarray:
    """
    Compute a local "gradient" vector on the current GPU.

    For demo purposes, we initialize:
        grad[i] = rank + 0.001 * i

    Parameters
    ----------
    num_elements : int
        Length of gradient vector.
    device : Device
        CUDA device object.
    stream : Stream
        CUDA stream object (created at device initialization).
    rank : int
        MPI rank ID.

    Returns
    -------
    cupy.ndarray
        Gradient vector on GPU.
    """
    # Create gradient array (CuPy uses the stream set at device initialization)
    grad = cp.empty(num_elements, dtype=cp.float32)

    # Use a CUDA kernel compiled with cuda.core to fill the array
    kernel = get_init_kernel(device)

    threads_per_block = 256
    blocks_per_grid = (num_elements + threads_per_block - 1) // threads_per_block
    config = LaunchConfig(grid=blocks_per_grid, block=threads_per_block)

    # Launch kernel using cuda.core stream
    launch(stream, config, kernel, grad.data.ptr, num_elements, rank)

    return grad


# ============================================================================
# MPI Allreduce to average gradients (host-staging)
# ============================================================================


def average_gradients(
    local_grad: cp.ndarray, comm: object, world_size: int
) -> cp.ndarray:
    """
    Average gradients across all MPI ranks using host-staging Allreduce.

    Steps:
    1. Copy local gradients from GPU to CPU (NumPy).
    2. Perform MPI_Allreduce on host buffers.
    3. Divide by world_size to obtain the average.
    4. Copy the averaged gradients back to GPU.

    This pattern is environment-agnostic and works on any MPI stack.
    """
    assert local_grad.dtype == cp.float32

    # GPU -> CPU
    local_host = local_grad.get()  # NumPy array on host
    avg_host = local_host.copy()

    # Allreduce on host buffers
    comm.Allreduce(local_host, avg_host, op=MPI.SUM)

    # Average
    avg_host /= world_size

    # CPU -> GPU
    avg_grad = cp.asarray(avg_host)

    return avg_grad


# ============================================================================
# Testing and verification
# ============================================================================


def main():
    """Demo: Multi-GPU gradient averaging with MPI (host-staging Allreduce)."""
    import argparse

    # Initialize MPI
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser(
        description=(
            "Multi-GPU Gradient Average with mpi4py + cuda.core "
            "(host-staging Allreduce)"
        )
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Number of gradient elements per GPU (default: 1024)",
    )
    args = parser.parse_args()

    num_elements = args.size

    # Initialize device and stream
    device = None
    stream = None
    try:
        device, stream = init_device(rank)

        if rank == 0:
            print(f"[Rank 0] World size = {world_size}")
        comm.Barrier()

        # Validate world size
        if world_size < 2:
            if rank == 0:
                print("=" * 70)
                print("ERROR: This sample requires at least 2 MPI processes!")
                print("=" * 70)
                print("\nPlease run with mpirun:")
                print("  mpirun -np 2 python multiGPUGradientAverage.py")
                print("  mpirun -np 4 python multiGPUGradientAverage.py --size 10000")
                print("\nFor multi-GPU systems:")
                print("  mpirun -np N python multiGPUGradientAverage.py")
                print("  (where N = number of GPUs)")
                print("=" * 70)
            sys.exit(1)

        # Validate input
        if num_elements <= 0:
            if rank == 0:
                print("Error: --size must be positive")
            sys.exit(1)

        if rank == 0:
            print("\n" + "=" * 70)
            print("Multi-GPU Gradient Average Demo")
            print("=" * 70)
            print(f"Number of MPI ranks (GPUs): {world_size}")
            print(f"Gradient vector length per GPU: {num_elements}")
            print(f"Device: {device.name}")
            print("Computation: gradients computed on GPU via cuda.core.")
            print(
                "Communication: gradients averaged via MPI_Allreduce on host "
                "(CPU) buffers."
            )
            print("=" * 70)

        # Step 1: Compute local gradients on each GPU
        # Use cuda.core Event for GPU timing measurements
        timing_options = EventOptions(enable_timing=True)
        start_event = stream.record(options=timing_options)

        local_grad = compute_local_gradients(num_elements, device, stream, rank)

        # Record end event and synchronize to ensure timing is complete
        end_event = stream.record(options=timing_options)
        end_event.sync()

        # Calculate elapsed time: Event subtraction returns milliseconds
        kernel_time = end_event - start_event

        # Step 2: Average gradients across all ranks (host-staging Allreduce)
        # Use CPU timing for MPI communication (host-staging includes GPU↔CPU transfers)
        import time

        comm_start = time.time()
        avg_grad = average_gradients(local_grad, comm, world_size)
        comm_time = (time.time() - comm_start) * 1000  # Convert to ms

        # Step 3: Sanity check on rank 0
        # For each element i:
        #   local_grad_r[i] = r + 0.001 * i, r = 0..world_size-1
        # Sum over ranks:
        #   sum[i] = sum_r r + 0.001 * i * world_size
        # Average:
        #   avg[i] = (0 + ... + (world_size-1))/world_size + 0.001 * i
        #          = (world_size - 1)/2 + 0.001 * i
        #
        # We verify this formula.

        expected_base = (world_size - 1) / 2.0
        i0 = 0
        i1 = num_elements // 2
        i2 = num_elements - 1

        # Copy a few sample elements back to host for printing on rank 0
        if rank == 0:
            avg_host_samples = avg_grad[[i0, i1, i2]].get()
            print("\nSample averaged gradient values (rank 0):")
            print(f"  avg_grad[{i0}] = {avg_host_samples[0]:.6f}")
            print(f"  avg_grad[{i1}] = {avg_host_samples[1]:.6f}")
            print(f"  avg_grad[{i2}] = {avg_host_samples[2]:.6f}")

            expected0 = expected_base + 0.001 * i0
            expected1 = expected_base + 0.001 * i1
            expected2 = expected_base + 0.001 * i2
            print("\nExpected values:")
            print(f"  expected[{i0}] = {expected0:.6f}")
            print(f"  expected[{i1}] = {expected1:.6f}")
            print(f"  expected[{i2}] = {expected2:.6f}")

        # All ranks perform a full-array correctness check on GPU
        expected_full = expected_base + 0.001 * cp.arange(
            num_elements, dtype=cp.float32
        )

        # Use utility function to verify results
        if rank == 0:
            print("\nVerifying gradient averaging correctness...")
        ok = verify_array_result(
            avg_grad, expected_full, rtol=1e-5, atol=1e-8, verbose=(rank == 0)
        )

        # Ensure all ranks agree on correctness
        ok_all = comm.allreduce(ok, op=MPI.LAND)

        if rank == 0:
            if ok_all:
                print("[PASS] Gradient averaging is correct on all ranks.")
            else:
                print(
                    "[FAIL] Gradient averaging mismatch detected on one or more ranks."
                )

            print("\nPerformance:")
            print(f"  Kernel time (GPU only): {kernel_time:.3f} ms")
            print(
                "  MPI communication time (host-staging, end-to-end): "
                f"{comm_time:.3f} ms"
            )
            print(f"  Total time: {kernel_time + comm_time:.3f} ms")

            print("\n" + "=" * 70)
            print("Demo complete.")
            print("=" * 70)

        return 0 if ok_all else 1
    finally:
        # Clean up stream resources
        if stream is not None:
            stream.close()
            cp.cuda.Stream.null.use()  # Reset CuPy's current stream to the null stream


if __name__ == "__main__":
    sys.exit(main())
