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
Single-Pass Multi-Block Reduction with Cooperative Groups

Demonstrates single-kernel multi-stage reduction using grid-wide
synchronization. Traditional reduction needs multiple kernel launches,
but with grid.sync() from Cooperative Groups, we can complete all
stages in ONE kernel.

Key Features:
- Grid-wide synchronization with grid.sync()
- Two-stage reduction in a single kernel (no atomic operations)
- Requires compute capability 6.0+ and cooperative launch
- Achieves 400-700 GB/s on modern GPUs

How it differs from other samples:
- blockArraySum.py: Basic thread/block indexing + atomicAdd
- reduction.py: High-performance shared memory, two-kernel approach
- This sample: Single-kernel multi-stage with grid.sync()

Transfers use CuPy on the same CUDA stream as ``launch()`` (``Stream.from_external``),
not ``cuda.bindings.driver`` memcpy. GPU timing uses CUDA events.
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
utilities_path = os.path.join(os.path.dirname(__file__), "..", "..", "Utilities")
sys.path.insert(0, utilities_path)
from cuda_samples_utils import verify_array_result  # noqa: E402


def _validate_threads_arg(threads):
    if threads is None:
        return None
    if threads <= 0 or threads > 1024:
        return "threads must be between 1 and 1024"
    if (threads & (threads - 1)) != 0:
        return (
            "threads must be a power of 2 "
            "(required by the shared-memory tree reduction kernel)"
        )
    return None


# Single-pass multi-block reduction kernel with grid-wide sync
REDUCTION_KERNEL = """
/*
 * Single-Kernel Multi-Stage Reduction using grid.sync()
 *
 * Strategy:
 *   Stage 1: Each block reduces its portion → partial sum
 *   grid.sync() ← KEY: All blocks synchronize
 *   Stage 2: Block 0 reduces all partial sums → final result
 *
 * Key feature: grid.sync() enables multi-stage within ONE kernel
 */

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
extern "C" __global__ void reduceSinglePassMultiBlockCG(
    const float *__restrict__ g_idata,
    float *__restrict__ g_odata,
    unsigned int n)
{
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;

    // Stage 1: Grid-stride loop + block reduction
    float sum = 0.0f;
    for (unsigned int i = grid.thread_rank(); i < n; i += grid.size()) {
        sum += g_idata[i];
    }

    sdata[tid] = sum;
    cg::sync(cta);

    // Block reduction (sequential addressing)
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        cg::sync(cta);
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }

    // KEY: Grid-wide synchronization (all blocks wait here)
    grid.sync();

    // Stage 2: Block 0 reduces all partial sums → final result
    // Use a stride loop so all gridDim.x partial sums are covered even
    // when gridDim.x > blockDim.x.
    if (blockIdx.x == 0) {
        // mySum stays 0.0f when tid >= gridDim.x (loop never executes),
        // implicitly zero-filling sdata for threads beyond the partial-sum count.
        float mySum = 0.0f;
        for (unsigned int i = tid; i < gridDim.x; i += blockSize) {
            mySum += g_odata[i];
        }
        sdata[tid] = mySum;
        cg::sync(cta);

        for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            cg::sync(cta);
        }

        if (tid == 0) {
            g_odata[0] = sdata[0];
        }
    }
}
"""


def get_max_cooperative_blocks(device, kernel, threads_per_block, shared_mem_bytes):
    """
    Calculate max blocks for cooperative launch (all must be resident).

    This is a conservative estimate that ignores shared memory limits;
    for precise tuning, use cudaOccupancyMaxActiveBlocksPerMultiprocessor.
    """
    # Get device properties
    prop = device.properties

    # Calculate maximum blocks per SM
    # Note: We use cudaOccupancyMaxActiveBlocksPerMultiprocessor functionality
    # For simplicity in Python, we'll use a conservative estimate
    num_sms = prop.multiprocessor_count
    max_threads_per_sm = prop.max_threads_per_multiprocessor
    max_blocks_per_sm = max_threads_per_sm // threads_per_block

    # Total blocks = blocks per SM × number of SMs
    max_blocks = max_blocks_per_sm * num_sms

    # Also respect max_grid_dim_x
    max_blocks = min(max_blocks, prop.max_grid_dim_x)

    return max_blocks


def run(
    num_elements=1 << 25,
    max_threads=None,
    max_blocks=None,
    test_iterations=100,
    cuda_include_dir=None,
):
    """Run single-pass multi-block reduction benchmark."""

    if cuda_include_dir is None:
        raise ValueError("cuda_include_dir is required")

    print("\n" + "=" * 70)
    print("Single-Pass Multi-Block Reduction with Cooperative Groups")
    print("=" * 70)
    msg = "Multi-stage reduction in a single kernel using grid.sync()"
    print(f"\nDemonstrates: {msg}")

    # Initialize device
    device = Device()
    device.set_current()
    major, minor = device.compute_capability

    print("\nDevice Information:")
    print(f"  Name: {device.name}")
    print(f"  Compute Capability: sm_{major}.{minor}")

    # Get device properties for configuration
    prop = device.properties

    # Determine threads per block
    if max_threads is None:
        max_threads = prop.max_threads_per_block
    threads_per_block = min(max_threads, 1024)

    # Define data type and itemsize
    itemsize = np.dtype(np.float32).itemsize

    print("\nReduction Configuration:")
    print(f"  Number of elements: {num_elements:,}")
    print(f"  Data size: {num_elements * itemsize / (1024**2):.2f} MB")

    # Compile kernel
    print("\nCompiling CUDA kernel...")
    # Support colon-separated multiple include paths
    include_paths = cuda_include_dir.split(":")
    program_options = ProgramOptions(
        std="c++17", arch=f"sm_{device.arch}", include_path=include_paths
    )
    prog = Program(REDUCTION_KERNEL, code_type="c++", options=program_options)
    mod = prog.compile("cubin")
    kernel = mod.get_kernel("reduceSinglePassMultiBlockCG")
    print("  Kernel compiled successfully")

    # Calculate blocks for cooperative launch
    shared_mem_bytes = threads_per_block * itemsize

    if max_blocks is None:
        max_blocks = get_max_cooperative_blocks(
            device, kernel, threads_per_block, shared_mem_bytes
        )

    # Calculate optimal blocks (all must be resident)
    num_blocks = min(
        max_blocks, (num_elements + threads_per_block - 1) // threads_per_block
    )

    print("\nLaunch Configuration:")
    print(f"  Threads per block: {threads_per_block}")
    print(f"  Number of blocks: {num_blocks}")
    print(f"  Total threads: {num_blocks * threads_per_block:,}")
    print(f"  Shared memory per block: {shared_mem_bytes} bytes")
    print("  Launch mode: Cooperative (grid-wide sync enabled)")

    # Generate random input data
    print("\n> Generating random input data...")
    rng = np.random.default_rng(42)
    h_idata = (rng.random(num_elements) * 256).astype(np.float32)

    stream = device.create_stream()
    cp_stream = cp.cuda.Stream.from_external(stream)
    try:
        d_odata = cp.empty(num_blocks, dtype=np.float32)
        with cp_stream:
            d_idata = cp.asarray(h_idata, dtype=np.float32)
        stream.sync()

        # Compute CPU reference
        print("> Computing reference result on CPU...")
        cpu_start = time.perf_counter()
        cpu_result = float(np.sum(h_idata))
        cpu_time = time.perf_counter() - cpu_start
        print(f"  CPU time: {cpu_time:.6f} seconds")

        # Warm-up
        print("\n> Warming up GPU...")

        launch_config = LaunchConfig(
            grid=(num_blocks, 1, 1),
            block=(threads_per_block, 1, 1),
            shmem_size=shared_mem_bytes,
            cooperative_launch=True,
        )

        n_u32 = np.uint32(num_elements)
        ptr_in = d_idata.data.ptr
        ptr_out = d_odata.data.ptr

        try:
            launch(stream, launch_config, kernel, ptr_in, ptr_out, n_u32)
        except Exception as e:
            print(f"  Cooperative launch failed: {e}")
            return 1

        stream.sync()
        print("  Warm-up successful")

        # Benchmark (CUDA events — not host wall clock around the whole loop)
        print(f"\n> Running benchmark ({test_iterations} iterations)...")
        event_options = EventOptions(enable_timing=True)
        start_event = stream.device.create_event(options=event_options)
        end_event = stream.device.create_event(options=event_options)
        # cuda.core event elapsed time (end - start) is in milliseconds (CUDA API).
        gpu_times_ms = []
        for _ in range(test_iterations):
            try:
                stream.record(start_event)
                launch(stream, launch_config, kernel, ptr_in, ptr_out, n_u32)
                stream.record(end_event)
                end_event.sync()
                gpu_times_ms.append(float(end_event - start_event))
            except Exception as e:
                print(f"Benchmark iteration failed: {e}")
                return 1

        avg_gpu_ms = float(np.mean(gpu_times_ms))
        avg_gpu_s = avg_gpu_ms / 1000.0

        stream.sync()
        with cp_stream:
            h_result = cp.asnumpy(d_odata[:1])
        gpu_result = float(h_result[0])

        # Performance metrics use seconds for throughput and speedup.
        # CPU time is already in seconds.
        bytes_processed = num_elements * 4
        throughput_gb_s = bytes_processed / avg_gpu_s / 1e9

        print("\n> Performance Results:")
        print(f"  Average GPU time: {avg_gpu_ms:.6f} ms")
        print(f"  Throughput: {throughput_gb_s:.2f} GB/s")
        print(f"  Speedup vs CPU: {cpu_time / avg_gpu_s:.2f}x")

        # Validate results
        print("\n> Validating results...")
        success = verify_array_result(
            np.array([gpu_result]),
            np.array([cpu_result]),
            rtol=1e-5,
            atol=1e-5,
        )

        # Summary
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"""
Single-kernel two-stage reduction:
  Stage 1: {num_blocks} blocks → {num_blocks} partial sums
  grid.sync() ← All blocks synchronize (KEY innovation)
  Stage 2: Block 0 → 1 final result
  Total: 1 kernel launch, {throughput_gb_s:.2f} GB/s

Comparison:
  • Traditional: 2 kernel launches or kernel + CPU
  • This sample: 1 kernel with grid.sync() between stages
  • Benefit: Eliminates ~5-20μs launch overhead per stage
    """)

        print("=" * 70)
        if success:
            print("Single-Pass Multi-Block Reduction completed successfully!")
        else:
            print("Single-Pass Multi-Block Reduction FAILED!")
        print("=" * 70 + "\n")

        return 0 if success else 1
    finally:
        stream.close()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Single-Pass Multi-Block Reduction with Cooperative Groups"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=1 << 25,
        help="Number of elements to reduce (default: 33554432 = 2^25)",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "Threads per block, power of 2 in [1, 1024]; "
            "default: device maximum (typically 1024)"
        ),
    )

    parser.add_argument(
        "--maxblocks",
        type=int,
        default=None,
        help=(
            "Maximum number of blocks "
            "(default: auto-calculated for cooperative launch)"
        ),
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )

    parser.add_argument(
        "--cuda-include-dir",
        type=str,
        required=True,
        help=(
            "CUDA include directory for NVRTC "
            "(can use colon-separated paths, e.g., /path1:/path2)"
        ),
    )

    args = parser.parse_args()

    # Validate arguments
    if args.n <= 0:
        print("Error: n must be positive")
        return 1

    err_threads = _validate_threads_arg(args.threads)
    if err_threads:
        print(f"Error: {err_threads}")
        return 1

    if args.maxblocks is not None and args.maxblocks <= 0:
        print("Error: maxblocks must be positive")
        return 1

    try:
        exit_code = run(
            num_elements=args.n,
            max_threads=args.threads,
            max_blocks=args.maxblocks,
            test_iterations=args.iterations,
            cuda_include_dir=args.cuda_include_dir,
        )
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
