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
Launch Configuration Tuning

Demonstrates how to find the optimal threads-per-block configuration for CUDA
kernels using cuda.core APIs. Benchmarks different thread layouts to answer:
"What is the best threads-per-block for my kernel?"
"""

import sys

try:
    import numpy as np
    from cuda.core import (
        Device,
        EventOptions,
        LaunchConfig,
        ManagedMemoryResource,
        ManagedMemoryResourceOptions,
        Program,
        ProgramOptions,
        launch,
    )
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


# =============================================================================
# CUDA Kernel Source Code
# =============================================================================

# Vector Addition Kernel - Simple memory-bound kernel (grid-stride loop)
VECTOR_ADD_KERNEL = r"""
extern "C" __global__
void vector_add(const float* __restrict__ a,
                const float* __restrict__ b,
                float* __restrict__ c,
                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}
"""

# Reduction Kernel - Sensitive to block size due to shared memory (grid-stride load)
REDUCTION_KERNEL = r"""
extern "C" __global__
void reduce_sum(const float* __restrict__ input,
                float* __restrict__ partial_sums,
                int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Load data into shared memory (grid-stride loop)
    float sum = 0.0f;
    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        sum += input[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}
"""


# =============================================================================
# Utility Functions
# =============================================================================


def compile_kernel(device, kernel_code, kernel_name):
    """Compile a CUDA kernel using cuda.core.Program."""
    arch = f"sm_{device.arch}"
    options = ProgramOptions(arch=arch)
    program = Program(kernel_code, code_type="c++", options=options)
    compiled = program.compile(target_type="cubin")
    return compiled.get_kernel(kernel_name)


def benchmark_kernel_1d(
    device,
    stream,
    kernel,
    args,
    n_elements,
    block_size,
    n_iterations=100,
    shared_mem_bytes=0,
):
    """
    Benchmark a 1D kernel with given threads-per-block configuration.
    Uses CUDA events for accurate GPU timing.

    Returns timing statistics as a dictionary.
    """
    grid_size = (n_elements + block_size - 1) // block_size

    config = LaunchConfig(
        grid=(grid_size,), block=(block_size,), shmem_size=shared_mem_bytes
    )

    # Warm-up run
    launch(stream, config, kernel, *args)
    stream.sync()

    # Timed runs with CUDA events
    event_opts = EventOptions(enable_timing=True)
    start_event = device.create_event(options=event_opts)
    end_event = device.create_event(options=event_opts)

    stream.record(start_event)
    for _ in range(n_iterations):
        launch(stream, config, kernel, *args)
    stream.record(end_event)
    end_event.sync()

    elapsed_ms = (end_event - start_event) / n_iterations

    return {
        "block_size": block_size,
        "grid_size": grid_size,
        "mean_time_ms": elapsed_ms,
        "std_time_ms": 0.0,  # Single measurement with events
    }


def print_gpu_info(device):
    """Print GPU information relevant to launch configuration."""
    print(f"\nDevice: {device.name}")
    cc = device.compute_capability
    print(f"Compute Capability: {cc.major}.{cc.minor}")


def allocate_managed_array(mr, stream, n_elements, dtype=np.float32):
    """Allocate device-preferred unified memory and return buffer with numpy view."""
    n_bytes = n_elements * np.dtype(dtype).itemsize
    buffer = mr.allocate(n_bytes, stream)
    stream.sync()

    # Zero-copy numpy view via DLPack (holds reference to buffer)
    np_view = np.from_dlpack(buffer).view(dtype).reshape(n_elements)
    return buffer, np_view


# =============================================================================
# Benchmark Demonstrations
# =============================================================================


def demo_vector_add_tuning(device, stream, mr, kernel):
    """Demonstrate launch configuration tuning for vector addition."""
    print("\n" + "=" * 60)
    print("VECTOR ADDITION - Launch Configuration Tuning")
    print("=" * 60)

    N = 10_000_000  # 10 million elements
    print(f"\nProblem size: {N:,} elements")
    print("Kernel: vector_add (C = A + B)")

    # Allocate device-preferred unified memory via cuda.core
    d_a, np_a = allocate_managed_array(mr, stream, N)
    d_b, np_b = allocate_managed_array(mr, stream, N)
    d_c, np_c = allocate_managed_array(mr, stream, N)
    try:
        # Initialize data via numpy views
        np_a[:] = np.random.rand(N).astype(np.float32)
        np_b[:] = np.random.rand(N).astype(np.float32)
        stream.sync()

        # Thread configurations to test (multiples of warp size = 32)
        thread_configs = [32, 64, 128, 256, 512, 1024]

        print(f"\nTesting thread configurations: {thread_configs}")
        print("-" * 60)

        results = []
        for tpb in thread_configs:
            result = benchmark_kernel_1d(
                device,
                stream,
                kernel,
                (d_a, d_b, d_c, np.int32(N)),
                N,
                tpb,
                n_iterations=100,
            )
            results.append(result)
            print(
                f"Block Size: {tpb:4d} | Blocks: {result['grid_size']:6d} | "
                f"Time: {result['mean_time_ms']:.4f} ms"
            )

        # Find optimal and worst configurations
        best = min(results, key=lambda x: x["mean_time_ms"])
        worst = max(results, key=lambda x: x["mean_time_ms"])

        print("-" * 60)
        print(
            f"\n✓ OPTIMAL: block_size={best['block_size']} "
            f"({best['mean_time_ms']:.4f} ms)"
        )
        print(
            f"✗ WORST:   block_size={worst['block_size']} "
            f"({worst['mean_time_ms']:.4f} ms)"
        )
        print(f"  Speedup: {worst['mean_time_ms']/best['mean_time_ms']:.2f}x")

        # Verify result
        stream.sync()
        expected = np_a + np_b
        if np.allclose(np_c, expected):
            print("\n✓ Results verified correct!")

        return results
    finally:
        d_a.close()
        d_b.close()
        d_c.close()


def demo_reduction_tuning(device, stream, mr, kernel):
    """Demonstrate launch config tuning for reduction (shared memory)."""
    print("\n" + "=" * 60)
    print("REDUCTION - Launch Configuration Tuning")
    print("=" * 60)

    N = 16_777_216  # 16M elements (power of 2)

    print(f"\nProblem size: {N:,} elements")
    print("Kernel: reduce_sum (parallel reduction)")
    print("Note: Reduction uses shared memory - more sensitive to block size!")

    # Allocate device-preferred unified memory via cuda.core
    d_input, np_input = allocate_managed_array(mr, stream, N)
    try:
        np_input[:] = np.random.rand(N).astype(np.float32)
        stream.sync()

        thread_configs = [32, 64, 128, 256, 512, 1024]

        print(f"\nTesting thread configurations: {thread_configs}")
        print("-" * 60)

        results = []
        for tpb in thread_configs:
            # Allocate partial sums array
            n_blocks = (N + tpb - 1) // tpb
            d_partial, _ = allocate_managed_array(mr, stream, n_blocks)
            try:
                # Shared memory size = block_size * sizeof(float)
                shared_mem_bytes = tpb * 4

                result = benchmark_kernel_1d(
                    device,
                    stream,
                    kernel,
                    (d_input, d_partial, np.int32(N)),
                    N,
                    tpb,
                    n_iterations=50,
                    shared_mem_bytes=shared_mem_bytes,
                )
                results.append(result)
                print(
                    f"Block Size: {tpb:4d} | Blocks: {result['grid_size']:6d} | "
                    f"Time: {result['mean_time_ms']:.4f} ms"
                )
            finally:
                d_partial.close()

        best = min(results, key=lambda x: x["mean_time_ms"])
        worst = max(results, key=lambda x: x["mean_time_ms"])

        print("-" * 60)
        print(f"\n✓ OPTIMAL: block_size={best['block_size']}")
        print(
            f"  Speedup over worst: {worst['mean_time_ms']/best['mean_time_ms']:.2f}x"
        )

        return results
    finally:
        d_input.close()


# =============================================================================
# Main
# =============================================================================


def main():
    """
    Complete demonstration of CUDA launch configuration tuning.

    This sample shows:
    1. Device initialization with cuda.core.Device
    2. Kernel compilation with cuda.core.Program
    3. Benchmarking different thread block configurations
    4. Finding optimal threads-per-block for various kernel types
    """
    print("=" * 60)
    print("Launch Configuration Tuning (cuda.core)")
    print("Finding the Best Block Size for Your Kernel")
    print("=" * 60)

    # Initialize CUDA device
    device = Device(0)
    device.set_current()

    # Print GPU information
    print_gpu_info(device)

    # Create stream and device-preferred memory resource
    stream = device.create_stream()
    mr_options = ManagedMemoryResourceOptions(preferred_location=device.device_id)
    mr = ManagedMemoryResource(mr_options)

    try:
        # Compile kernels
        print("\nCompiling CUDA kernels with cuda.core.Program...")
        arch = f"sm_{device.arch}"
        print(f"  Target architecture: {arch}")

        vec_add_kernel = compile_kernel(device, VECTOR_ADD_KERNEL, "vector_add")
        print("  ✓ vector_add kernel compiled")

        reduction_kernel = compile_kernel(device, REDUCTION_KERNEL, "reduce_sum")
        print("  ✓ reduce_sum kernel compiled")

        # Run demonstrations
        demo_vector_add_tuning(device, stream, mr, vec_add_kernel)
        demo_reduction_tuning(device, stream, mr, reduction_kernel)

        print("\n" + "=" * 60)
        print("SAMPLE COMPLETE")
        print("=" * 60)
        print("\nKey Takeaway: The optimal thread configuration depends on your")
        print("specific kernel characteristics. Always benchmark to find the best!")
        print()
    finally:
        stream.close()


if __name__ == "__main__":
    main()
