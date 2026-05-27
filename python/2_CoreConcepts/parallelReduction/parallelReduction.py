# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
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
Parallel Reduction using cuda.core and cuda.compute

Demonstrates efficient parallel summation of large arrays on GPU:
1. Custom CUDA kernel showing reduction tree pattern and synchronization
2. cuda.compute.reduce_into() for production-ready reduction

Key Concepts:
- Reduction tree pattern: Divide-and-conquer parallel algorithm
- Thread synchronization: Using __syncthreads() for coordination
- Sequential thread IDs: How to avoid warp divergence
- cuda.core Stream integration with CuPy via ExternalStream
"""

import math
import sys
from pathlib import Path

# Add Utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))

try:
    import cupy as cp
    import numpy as np
    from cuda.compute import OpKind, reduce_into
    from cuda.core import (
        Device,
        Kernel,
        LaunchConfig,
        Program,
        ProgramOptions,
        Stream,
        launch,
    )
    from cuda_samples_utils import print_gpu_info, verify_array_result
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


# =============================================================================
# CUDA Kernel: Parallel Reduction (optimized - no warp divergence)
# =============================================================================
REDUCTION_KERNEL: str = r"""
extern "C" __global__
void reduce_sum(const float* __restrict__ input,
                float* __restrict__ output, int n) {
    /*
     * Parallel reduction using grid-stride loop (canonical pattern) and
     * sequential thread IDs for the reduction tree (avoids warp divergence).
     *
     * Grid-stride loop: each thread processes multiple elements
     *   for (i = tid; i < n; i += gridDim.x * blockDim.x)
     *
     * Reduction tree: sequential addressing keeps warps coherent.
     */
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int grid_stride = (unsigned int)gridDim.x * blockDim.x;

    float sum = 0.0f;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += grid_stride) {
        sum += input[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory (sequential addressing - no divergence)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // Wait for all threads before next iteration
    }

    // Thread 0 writes block result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
"""


def compile_kernel(device: Device) -> Kernel:
    """Compile the reduction kernel for the given device."""
    arch = f"sm_{device.arch}"
    options = ProgramOptions(arch=arch)
    program = Program(REDUCTION_KERNEL, code_type="c++", options=options)
    return program.compile(target_type="cubin").get_kernel("reduce_sum")


def reduction_stage_output_counts(n: int, block_size: int) -> list[int]:
    """Lengths of intermediate arrays for each multi-launch reduction stage."""
    counts: list[int] = []
    while n > 1:
        num_blocks = math.ceil(n / block_size)
        counts.append(num_blocks)
        n = num_blocks
    return counts


def reduce_custom(
    stream: Stream,
    kernel: Kernel,
    d_input: cp.ndarray,
    block_size: int = 256,
    sync: bool = True,
    work_buffers: list[cp.ndarray] | None = None,
) -> float | cp.ndarray:
    """
    Perform parallel reduction using custom CUDA kernel.

    Uses multiple kernel launches to reduce array to single value.
    Each launch reduces by factor of block_size.

    When sync=True (default), syncs and returns the scalar result.
    When sync=False, returns the 1-element array without syncing;
    caller must sync before reading (avoids host overhead in benchmarks).

    work_buffers: optional list of device arrays, one per stage, with length
    at least each stage's output count (from ``reduction_stage_output_counts``).
    When provided, avoids per-call allocation (e.g. for benchmarking).
    """
    n = len(d_input)
    current = d_input
    stage = 0

    if work_buffers is not None:
        expected_counts = reduction_stage_output_counts(n, block_size)
        if len(work_buffers) != len(expected_counts):
            msg = (
                f"work_buffers length {len(work_buffers)} != "
                f"{len(expected_counts)} stages"
            )
            raise ValueError(msg)

    while n > 1:
        num_blocks = math.ceil(n / block_size)
        if work_buffers is not None:
            d_output = work_buffers[stage]
            if d_output.size < num_blocks:
                msg = f"work_buffers[{stage}] size {d_output.size} < {num_blocks}"
                raise ValueError(msg)
            if d_output.size != num_blocks:
                d_output = d_output[:num_blocks]
        else:
            d_output = cp.empty(num_blocks, dtype=cp.float32)

        config = LaunchConfig(
            grid=(num_blocks, 1, 1),
            block=(block_size, 1, 1),
            shmem_size=block_size * 4,  # float = 4 bytes
        )

        launch(
            stream,
            config,
            kernel,
            current.data.ptr,
            d_output.data.ptr,
            np.int32(n),
        )

        current = d_output
        n = num_blocks
        stage += 1

    if sync:
        stream.sync()
        return float(current[0])
    return current


def benchmark_custom(
    stream: Stream,
    kernel: Kernel,
    d_input: cp.ndarray,
    num_runs: int = 10,
    block_size: int = 256,
) -> tuple[float, float]:
    """Benchmark custom reduction kernel using cuda.core events."""
    stage_counts = reduction_stage_output_counts(len(d_input), block_size)
    work_buffers = [cp.empty(c, dtype=cp.float32) for c in stage_counts]

    # Warmup run (with sync to get valid result)
    _ = reduce_custom(
        stream, kernel, d_input, block_size=block_size, work_buffers=work_buffers
    )

    event_opts = {"enable_timing": True}
    start_event = stream.device.create_event(options=event_opts)
    end_event = stream.device.create_event(options=event_opts)

    times: list[float] = []
    result = 0.0

    for _ in range(num_runs):
        stream.record(start_event)
        d_result = reduce_custom(
            stream,
            kernel,
            d_input,
            block_size=block_size,
            sync=False,
            work_buffers=work_buffers,
        )
        stream.record(end_event)
        end_event.sync()
        result = float(d_result[0])

        times.append(end_event - start_event)

    return result, float(np.mean(times))


def benchmark_cuda_compute(
    stream: Stream,
    d_input: cp.ndarray,
    num_runs: int = 10,
) -> tuple[float, float]:
    """Benchmark cuda.compute.reduce_into() using cuda.core events."""
    h_init = np.array([0.0], dtype=np.float32)

    # Warmup (includes JIT compilation)
    d_warmup = cp.empty(1, dtype=cp.float32)
    reduce_into(
        d_in=d_input,
        d_out=d_warmup,
        op=OpKind.PLUS,
        num_items=len(d_input),
        h_init=h_init,
        stream=stream,
    )
    stream.sync()

    d_output = cp.empty(1, dtype=cp.float32)
    event_opts = {"enable_timing": True}
    start_event = stream.device.create_event(options=event_opts)
    end_event = stream.device.create_event(options=event_opts)

    times: list[float] = []
    result = 0.0

    for _ in range(num_runs):
        stream.record(start_event)
        reduce_into(
            d_in=d_input,
            d_out=d_output,
            op=OpKind.PLUS,
            num_items=len(d_input),
            h_init=h_init,
            stream=stream,
        )
        stream.record(end_event)
        end_event.sync()

        result = float(d_output[0])
        times.append(end_event - start_event)

    return result, float(np.mean(times))


def main() -> bool:
    """Main function demonstrating parallel reduction."""
    print("=" * 70)
    print("Parallel Reduction - Efficient GPU Array Summation")
    print("=" * 70)

    device = Device(0)
    device.set_current()
    stream = device.create_stream()
    cp_stream = cp.cuda.ExternalStream(int(stream.handle))

    print()
    print_gpu_info(device)

    array_size = 1 << 20  # 1M elements
    h_input = np.random.rand(array_size).astype(np.float32)
    expected_sum = float(np.sum(h_input))

    print(f"\nArray size: {array_size:,} elements ({array_size * 4 / 1e6:.1f} MB)")
    print(f"Expected sum: {expected_sum:.6f}")

    print("\nCompiling custom CUDA kernel...")
    kernel = compile_kernel(device)

    try:
        with cp_stream:
            d_input = cp.asarray(h_input)

        # ======================================================================
        # Part 1: Custom Kernel
        # ======================================================================
        print("\n" + "=" * 70)
        print("PART 1: Custom Kernel (Educational)")
        print("=" * 70)

        result, time_ms = benchmark_custom(stream, kernel, d_input)

        print(f"\nReduction tree kernel:  {result:>14.2f}")
        print(f"Expected:               {expected_sum:>14.2f}")
        print(f"Time:                   {time_ms:>14.3f} ms")

        # ======================================================================
        # Part 2: cuda.compute (Production)
        # ======================================================================
        print("\n" + "=" * 70)
        print("PART 2: cuda.compute.reduce_into() (Production)")
        print("=" * 70)

        result_cc, time_cc = benchmark_cuda_compute(stream, d_input)

        print(f"\ncuda.compute result:    {result_cc:>14.2f}")
        print(f"Expected:               {expected_sum:>14.2f}")
        print(f"Time:                   {time_cc:>14.3f} ms")

        # Verify both results using principled rtol/atol
        with cp_stream:
            d_expected = cp.array([expected_sum], dtype=cp.float32)
            custom_ok = verify_array_result(
                cp.array([result], dtype=cp.float32),
                d_expected,
                rtol=1e-5,
                atol=1e-8,
                verbose=False,
            )
            compute_ok = verify_array_result(
                cp.array([result_cc], dtype=cp.float32),
                d_expected,
                rtol=1e-5,
                atol=1e-8,
                verbose=False,
            )
        if custom_ok and compute_ok:
            print("\nTest PASSED!")
            return True
        else:
            print("\nTest FAILED - Error too large!")
            return False
    finally:
        stream.close()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
