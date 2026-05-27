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
Fast Array Sum using Shared Memory - Two-Stage Reduction

Demonstrates efficient parallel reduction using shared memory and
two-stage approach to avoid atomic operation bottlenecks.

Key Features:
- Block-level reduction using shared memory
- Each thread loads 2 elements to reduce global memory traffic
- Sequential addressing tree reduction pattern
- No atomic operations - eliminates serialization bottleneck
- Device memory via CuPy; ``launch()`` takes pointers as ``ndarray.data.ptr``
- CuPy uses ``cp.cuda.Stream.from_external(stream)``.
"""

import argparse
import os
import sys
import time

try:
    import cupy as cp
    import numpy as np
    from cuda.core import (
        Device,
        EventOptions,
        LaunchConfig,
        Program,
        ProgramOptions,
        launch,
    )
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Import utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Utilities"))
from cuda_samples_utils import verify_array_result  # noqa: E402

# Two-stage block reduction kernel
REDUCTION_KERNEL = """
/*
 * Block-level reduction kernel using shared memory
 *
 * Strategy:
 *   - Each block processes blockSize * 2 elements
 *   - Uses shared memory for fast intra-block reduction
 *   - Outputs one partial sum per block (no atomics)
 *
 * Key optimizations:
 *   - Load 2 elements per thread (reduces global memory traffic by 50%)
 *   - Tree reduction with sequential addressing (avoids divergence)
 *   - Shared memory instead of atomic operations (eliminates bottleneck)
 *
 * Note: This sample provides separate implementations for each data type
 * for clarity. Production code typically uses templates with SharedMemory<T>
 * or reinterpret_cast to avoid duplication. See NVIDIA reduction guide for
 * template-based approaches.
 */

extern "C" __global__ void blockReduceKernel_int(
    const int *__restrict__ input,
    int *__restrict__ blockSums,
    unsigned int n)
{
    extern __shared__ int sdata_int[];

    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int gid = blockIdx.x * (blockSize * 2) + tid;

    // Load 2 elements per thread
    int sum = 0;
    if (gid < n) sum += input[gid];
    if (gid + blockSize < n) sum += input[gid + blockSize];

    sdata_int[tid] = sum;
    __syncthreads();

    // Tree reduction with sequential addressing
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_int[tid] += sdata_int[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        blockSums[blockIdx.x] = sdata_int[0];
    }
}

extern "C" __global__ void blockReduceKernel_float(
    const float *__restrict__ input,
    float *__restrict__ blockSums,
    unsigned int n)
{
    extern __shared__ float sdata_float[];

    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int gid = blockIdx.x * (blockSize * 2) + tid;

    // Load 2 elements per thread
    float sum = 0.0f;
    if (gid < n) sum += input[gid];
    if (gid + blockSize < n) sum += input[gid + blockSize];

    sdata_float[tid] = sum;
    __syncthreads();

    // Tree reduction with sequential addressing
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_float[tid] += sdata_float[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        blockSums[blockIdx.x] = sdata_float[0];
    }
}

extern "C" __global__ void blockReduceKernel_double(
    const double *__restrict__ input,
    double *__restrict__ blockSums,
    unsigned int n)
{
    extern __shared__ double sdata_double[];

    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int gid = blockIdx.x * (blockSize * 2) + tid;

    // Load 2 elements per thread
    double sum = 0.0;
    if (gid < n) sum += input[gid];
    if (gid + blockSize < n) sum += input[gid + blockSize];

    sdata_double[tid] = sum;
    __syncthreads();

    // Tree reduction with sequential addressing
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_double[tid] += sdata_double[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        blockSums[blockIdx.x] = sdata_double[0];
    }
}
"""


def reduce_cpu(data):
    """Compute sum using Kahan summation for numerical accuracy."""
    if len(data) == 0:
        return 0

    sum_val = float(data[0])
    c = 0.0

    for i in range(1, len(data)):
        y = float(data[i]) - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t

    return sum_val


def _validate_threads_per_block(threads_per_block):
    if threads_per_block <= 0 or threads_per_block > 1024:
        return "threads per block must be between 1 and 1024"
    if (threads_per_block & (threads_per_block - 1)) != 0:
        return (
            "threads per block must be a power of 2 "
            "(required by the shared-memory tree reduction kernel)"
        )
    return None


def run(
    num_elements=1 << 24, threads_per_block=256, test_iterations=100, datatype="float"
):
    """Run two-stage reduction benchmark."""

    print("\n" + "=" * 70)
    print("Fast Array Sum using Shared Memory - Two-Stage Reduction")
    print("=" * 70)
    print("\nDemonstrates: Efficient parallel reduction using shared memory")

    # Map datatype
    dtype_map = {"int": np.int32, "float": np.float32, "double": np.float64}
    if datatype not in dtype_map:
        print(f"Unknown datatype '{datatype}', using 'float'")
        datatype = "float"
    dtype = dtype_map[datatype]
    itemsize = np.dtype(dtype).itemsize

    # Initialize device
    device = Device()
    device.set_current()
    major, minor = device.compute_capability

    print("\nDevice Information:")
    print(f"  Name: {device.name}")
    print(f"  Compute Capability: sm_{major}.{minor}")

    # Configuration
    print("\nConfiguration:")
    print(f"  Array size: {num_elements:,} elements")
    print(f"  Data type: {datatype}")
    print(f"  Memory: {num_elements * itemsize / (1024**2):.2f} MB")
    print(f"  Threads per block: {threads_per_block}")

    # Calculate number of blocks
    # Each block processes threads_per_block * 2 elements
    num_blocks = (num_elements + threads_per_block * 2 - 1) // (threads_per_block * 2)

    print("\nTwo-Stage Reduction Strategy:")
    print("  Stage 1: GPU block reduction")
    print(f"    - Number of blocks: {num_blocks}")
    print(f"    - Elements per block: {threads_per_block * 2}")
    print(f"    - Output: {num_blocks} partial sums")
    print("  Stage 2: CPU final reduction")
    print(f"    - Combine {num_blocks} partial sums → 1 final result")

    # Compile kernel
    print("\nCompiling CUDA kernel...")
    program_options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
    prog = Program(REDUCTION_KERNEL, code_type="c++", options=program_options)
    mod = prog.compile("cubin")
    kernel_name = f"blockReduceKernel_{datatype}"
    kernel = mod.get_kernel(kernel_name)
    print(f"  Kernel '{kernel_name}' compiled successfully")

    # Generate input data
    print("\n> Generating random input data...")
    rng = np.random.default_rng(42)
    if datatype == "int":
        h_input = rng.integers(0, 256, size=num_elements, dtype=dtype)
    else:
        h_input = (rng.random(num_elements) * 256).astype(dtype)

    # cuda.core stream for launch/events; CuPy copies use the same stream via
    # Stream.from_external.
    stream = device.create_stream()
    cp_stream = cp.cuda.Stream.from_external(stream)
    try:
        d_blockSums = cp.empty(num_blocks, dtype=dtype)
        with cp_stream:
            d_input = cp.asarray(h_input, dtype=dtype)
        stream.sync()

        # Compute CPU reference
        print("> Computing reference result on CPU...")
        cpu_start = time.perf_counter()
        cpu_result = reduce_cpu(h_input)
        cpu_time = time.perf_counter() - cpu_start
        print(f"  CPU time: {cpu_time:.6f} seconds")

        # Configure launch
        shared_mem_bytes = threads_per_block * itemsize
        config = LaunchConfig(
            grid=num_blocks, block=threads_per_block, shmem_size=shared_mem_bytes
        )

        # Warm-up
        print("\n> Warming up GPU...")
        launch(
            stream,
            config,
            kernel,
            d_input.data.ptr,
            d_blockSums.data.ptr,
            np.uint32(num_elements),
        )
        stream.sync()
        print("  Warm-up completed")

        # Benchmark Stage 1 (GPU)
        print("\n> Benchmarking Stage 1 (GPU block reduction)...")
        print(f"  Running {test_iterations} iterations...")

        # cuda.core event elapsed time (end - start) is in milliseconds (CUDA API).
        stage1_times_ms = []
        event_options = EventOptions(enable_timing=True)
        start_event = stream.device.create_event(options=event_options)
        end_event = stream.device.create_event(options=event_options)
        for _ in range(test_iterations):
            stream.record(start_event)
            launch(
                stream,
                config,
                kernel,
                d_input.data.ptr,
                d_blockSums.data.ptr,
                np.uint32(num_elements),
            )
            stream.record(end_event)
            end_event.sync()
            stage1_times_ms.append(float(end_event - start_event))

        avg_stage1_ms = np.mean(stage1_times_ms)
        avg_stage1_s = avg_stage1_ms / 1000.0

        # Stage 2 (CPU)
        print("\n> Running Stage 2 (CPU final reduction)...")
        # Device → Host: after stream sync, partial sums are visible on host.
        stream.sync()
        with cp_stream:
            h_blockSums = cp.asnumpy(d_blockSums)
        stage2_start = time.perf_counter()
        gpu_result = float(np.sum(h_blockSums))
        stage2_time = time.perf_counter() - stage2_start

        total_time = avg_stage1_s + stage2_time

        # Performance metrics (use seconds for throughput; CPU times are in seconds)
        bytes_processed = num_elements * itemsize
        throughput = bytes_processed / avg_stage1_s / 1e9

        print("\n" + "=" * 70)
        print("Performance Results")
        print("=" * 70)
        print("\nStage 1 (GPU block reduction):")
        print(f"  Average time: {avg_stage1_ms:.6f} ms")
        print(f"  Throughput: {throughput:.2f} GB/s")
        print("\nStage 2 (CPU final reduction):")
        print(f"  Time: {stage2_time * 1000:.6f} ms")
        print(f"  ({num_blocks} partial sums)")
        print(f"\nTotal time: {total_time * 1000:.6f} ms")
        print(f"Speedup vs CPU: {cpu_time / total_time:.2f}x")

        # Validation
        print("\n> Validating results...")
        if datatype == "int":
            print(f"  GPU result: {int(gpu_result):,}")
            print(f"  CPU result: {int(cpu_result):,}")
            rtol, atol = 0.0, 0.0
        else:
            precision = 8 if datatype == "float" else 12
            print(f"  GPU result: {gpu_result:.{precision}f}")
            print(f"  CPU result: {cpu_result:.{precision}f}")
            if datatype == "float":
                rtol, atol = 1e-5, 1e-8 * num_elements
            else:  # double
                rtol, atol = 1e-8, 1e-12 * num_elements

        success = verify_array_result(
            np.array([gpu_result]),
            np.array([cpu_result]),
            rtol=rtol,
            atol=atol,
            verbose=True,
        )

        # Summary
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print("Key optimizations:")
        half_reads = num_elements // 2
        print(
            "  - Load 2 elements per thread: "
            f"{half_reads:,} global reads (50% savings)"
        )
        print("  - Shared memory for reduction: ~10-20x faster than global memory")
        print(f"  - Parallel block outputs: {num_blocks} independent writes")
        print(f"Result: {throughput:.2f} GB/s throughput")

        print("=" * 70)
        if success:
            print("Two-Stage Reduction completed successfully!")
        else:
            print("Two-Stage Reduction FAILED!")
        print("=" * 70 + "\n")

        return 0 if success else 1
    finally:
        stream.close()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Two-Stage Reduction with Shared Memory",
        epilog="See README.md for usage examples and detailed documentation.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=1 << 24,
        help="Number of elements to reduce (default: 16777216 = 2^24)",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=256,
        help="Threads per block, power of 2 in [1, 1024] (default: 256)",
    )

    parser.add_argument(
        "--type",
        type=str,
        default="float",
        choices=["int", "float", "double"],
        help="Data type for reduction (default: float)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.n <= 0:
        print("Error: n must be positive")
        return 1

    err = _validate_threads_per_block(args.threads)
    if err:
        print(f"Error: {err}")
        return 1

    try:
        exit_code = run(
            num_elements=args.n,
            threads_per_block=args.threads,
            test_iterations=args.iterations,
            datatype=args.type,
        )
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
