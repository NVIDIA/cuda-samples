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
Parallel Histogram with Atomics using cuda.core

This sample demonstrates GPU histogram computation using atomic operations,
showcasing the modern cuda.core API for:
- Kernel compilation (Program, ProgramOptions)
- Kernel launch configuration (LaunchConfig)
- Stream management (Stream)
- Event timing (EventOptions)

Two histogram approaches are compared:
1. Global Atomics - All threads atomically update global memory
2. Privatized Histograms - Shared memory reduces global atomic contention
"""

import sys

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
    print("Please install: pip install -r requirements.txt")
    sys.exit(1)


NUM_BINS = 256

# CUDA C source code for both histogram kernels
HISTOGRAM_KERNELS = r"""
// Global Atomics - simple but high contention on popular bins
extern "C" __global__
void histogram_global(const unsigned char* data, unsigned int* histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        atomicAdd(&histogram[data[i]], 1);
    }
}

// Privatized - uses shared memory to reduce global atomic contention
extern "C" __global__
void histogram_privatized(const unsigned char* data, unsigned int* histogram, int n) {
    __shared__ unsigned int local_hist[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize shared memory
    for (int i = tid; i < 256; i += blockDim.x)
        local_hist[i] = 0;
    __syncthreads();

    // Accumulate into shared memory (fast)
    for (int i = idx; i < n; i += stride)
        atomicAdd(&local_hist[data[i]], 1);
    __syncthreads();

    // Merge to global (fewer atomics)
    for (int i = tid; i < 256; i += blockDim.x)
        if (local_hist[i] > 0)
            atomicAdd(&histogram[i], local_hist[i]);
}
"""


def main():
    print("=" * 60)
    print("Parallel Histogram with Atomics (cuda.core)")
    print("=" * 60)

    # Initialize device using cuda.core
    device = Device(0)
    device.set_current()
    print(f"\nDevice: {device.name}")
    print(f"Compute Capability: {device.compute_capability}")

    # Create stream using cuda.core
    stream = device.create_stream()

    # Make CuPy use the same stream for correct ordering (avoids null-stream sync)
    cp.cuda.Stream.from_external(stream).use()

    try:
        _run_histogram(device, stream)
    finally:
        cp.cuda.Stream.null.use()  # Restore CuPy to default stream before closing
        stream.close()


def _run_histogram(device, stream):
    """Run histogram computation and benchmarking."""
    # Compile CUDA kernels using cuda.core.Program
    print("\nCompiling CUDA kernels with cuda.core.Program...")
    arch = f"sm_{device.arch}"
    options = ProgramOptions(arch=arch)
    program = Program(HISTOGRAM_KERNELS, code_type="c++", options=options)
    object_code = program.compile("cubin")

    kernel_global = object_code.get_kernel("histogram_global")
    kernel_privatized = object_code.get_kernel("histogram_privatized")
    print(f"  Compiled for architecture: {arch}")

    # Generate test data directly on GPU (more efficient than CPU->GPU copy)
    n = 10_000_000
    print(f"\nGenerating {n:,} random values on GPU...")
    data_gpu = cp.random.randint(0, 256, size=n, dtype=cp.uint8)
    hist_gpu = cp.zeros(NUM_BINS, dtype=cp.uint32)

    # Compute reference histogram on CPU for verification
    data_cpu = cp.asnumpy(data_gpu)
    hist_cpu, _ = np.histogram(data_cpu, bins=NUM_BINS, range=(0, 256))
    hist_cpu = hist_cpu.astype(np.uint32)

    # Configure kernel launch using cuda.core.LaunchConfig
    block_size = 256
    grid_size = min((n + block_size - 1) // block_size, 1024)
    config = LaunchConfig(grid=(grid_size,), block=(block_size,))

    print("\nVerifying correctness...")

    # Ensure CuPy allocations complete before kernel launch on our stream
    stream.sync()

    # Launch global atomics kernel (hist_gpu is already zeros from cp.zeros)
    launch(
        stream, config, kernel_global, data_gpu.data.ptr, hist_gpu.data.ptr, np.int32(n)
    )
    stream.sync()

    hist_global = cp.asnumpy(hist_gpu)
    global_ok = np.array_equal(hist_cpu, hist_global)
    print(f"  Global atomics:     {'PASSED' if global_ok else 'FAILED'}")

    # Reset histogram and launch privatized kernel (fill on same stream)
    hist_gpu.fill(0)
    launch(
        stream,
        config,
        kernel_privatized,
        data_gpu.data.ptr,
        hist_gpu.data.ptr,
        np.int32(n),
    )
    stream.sync()

    hist_privatized = cp.asnumpy(hist_gpu)
    privatized_ok = np.array_equal(hist_cpu, hist_privatized)
    print(f"  Privatized atomics: {'PASSED' if privatized_ok else 'FAILED'}")

    if not (global_ok and privatized_ok):
        sys.exit(1)

    # Benchmark using cuda.core Events (explicit Event objects recorded on stream)
    print("\nBenchmarking (100 iterations)...")
    num_iterations = 100
    event_opts = EventOptions(enable_timing=True)
    start_event = device.create_event(options=event_opts)
    end_event = device.create_event(options=event_opts)

    # Benchmark global atomics
    stream.record(start_event)
    for _ in range(num_iterations):
        hist_gpu.fill(0)
        launch(
            stream,
            config,
            kernel_global,
            data_gpu.data.ptr,
            hist_gpu.data.ptr,
            np.int32(n),
        )
    stream.record(end_event)
    end_event.sync()
    time_global = (end_event - start_event) / num_iterations

    # Benchmark privatized
    stream.record(start_event)
    for _ in range(num_iterations):
        hist_gpu.fill(0)
        launch(
            stream,
            config,
            kernel_privatized,
            data_gpu.data.ptr,
            hist_gpu.data.ptr,
            np.int32(n),
        )
    stream.record(end_event)
    end_event.sync()
    time_privatized = (end_event - start_event) / num_iterations

    print(f"  Global atomics:     {time_global:.3f} ms")
    print(f"  Privatized atomics: {time_privatized:.3f} ms")
    print(f"  Speedup:            {time_global / time_privatized:.1f}x")

    print("\nTest PASSED")


if __name__ == "__main__":
    main()
