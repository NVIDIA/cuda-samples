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
Block-wise Array Sum with Threaded Access

Demonstrates thread/block indexing, strided loops, and block-wise reduction.

Key Concepts:
    Global Thread ID = blockIdx.x * blockDim.x + threadIdx.x
    Stride = blockDim.x * gridDim.x
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))
from cuda_samples_utils import verify_array_result

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
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)


KERNELS_CODE: str = r"""
// Each thread processes one element
extern "C" __global__
void simple_indexing(const float* input, float* output, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        output[tid] = input[tid] * 2.0f;
    }
}

// Each thread processes multiple elements via strided access
extern "C" __global__
void strided_loop(const float* input, float* output, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = tid; i < N; i += stride) {
        output[i] = input[i] * 2.0f;
    }
}

// Block-wise partial sum with shared memory reduction
extern "C" __global__
void block_partial_sum(const float* input, float* partial_sums, size_t N) {
    extern __shared__ float sdata[];

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_tid = threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    // Each thread accumulates multiple elements (strided)
    float sum = 0.0f;
    for (size_t i = tid; i < N; i += stride) {
        sum += input[i];
    }
    sdata[local_tid] = sum;
    __syncthreads();

    // Block-level tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_tid < s) {
            sdata[local_tid] += sdata[local_tid + s];
        }
        __syncthreads();
    }

    if (local_tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}
"""


def run_sample(num_elements: int = 1024 * 1024, device_id: int = 0) -> bool:
    """
    Run block-wise sum demonstration.

    Parameters
    ----------
    num_elements : int
        Number of array elements
    device_id : int
        CUDA device ID

    Returns
    -------
    bool
        True if all tests passed
    """
    threads_per_block = 256
    num_blocks = 64

    device = Device(device_id)
    device.set_current()
    stream = device.create_stream()

    arch = f"sm_{device.arch}"
    print(f"Device: {device.name}")
    print(f"Compute Capability: {arch}")
    print(f"Array size: {num_elements:,} elements\n")

    try:
        # Make CuPy use our stream
        cp.cuda.ExternalStream(int(stream.handle)).use()

        # Compile kernels
        program = Program(
            KERNELS_CODE, code_type="c++", options=ProgramOptions(arch=arch)
        )
        module = program.compile(target_type="cubin")
        kernel_simple = module.get_kernel("simple_indexing")
        kernel_strided = module.get_kernel("strided_loop")
        kernel_sum = module.get_kernel("block_partial_sum")

        # Test data
        h_input = np.arange(num_elements, dtype=np.float32)
        d_input = cp.asarray(h_input)
        d_output = cp.zeros_like(d_input)
        expected = cp.asarray(h_input * 2.0)

        # Demo 1: Simple indexing (1 thread = 1 element)
        full_blocks = (num_elements + threads_per_block - 1) // threads_per_block
        config = LaunchConfig(grid=full_blocks, block=threads_per_block)
        launch(
            stream,
            config,
            kernel_simple,
            d_input.data.ptr,
            d_output.data.ptr,
            cp.uint64(num_elements),
        )
        stream.sync()
        print("Simple indexing: ", end="")
        test1 = verify_array_result(d_output, expected)

        # Demo 2: Strided loop (threads process multiple elements)
        d_output.fill(0)
        config = LaunchConfig(grid=num_blocks, block=threads_per_block)
        launch(
            stream,
            config,
            kernel_strided,
            d_input.data.ptr,
            d_output.data.ptr,
            cp.uint64(num_elements),
        )
        stream.sync()
        print("Strided loop:    ", end="")
        test2 = verify_array_result(d_output, expected)

        # Demo 3: Block-wise sum with shared memory
        d_ones = cp.ones(num_elements, dtype=cp.float32)
        d_partial = cp.zeros(num_blocks, dtype=cp.float32)
        shared_mem = threads_per_block * 4

        config = LaunchConfig(
            grid=num_blocks, block=threads_per_block, shmem_size=shared_mem
        )
        launch(
            stream,
            config,
            kernel_sum,
            d_ones.data.ptr,
            d_partial.data.ptr,
            cp.uint64(num_elements),
        )
        stream.sync()

        # Each block sums num_elements/num_blocks elements (strided access).
        # Requires num_elements % num_blocks == 0 for correct expected values.
        assert (
            num_elements % num_blocks == 0
        ), "num_elements must be divisible by num_blocks for block_partial_sum"
        expected_partial = cp.full(
            num_blocks, num_elements / num_blocks, dtype=cp.float32
        )
        print("Block-wise sum:  ", end="")
        test3 = verify_array_result(d_partial, expected_partial)

        # Performance timing
        event_opts = EventOptions(enable_timing=True)
        iterations = 100

        stream.sync()
        start = stream.record(options=event_opts)
        for _ in range(iterations):
            launch(
                stream,
                config,
                kernel_sum,
                d_ones.data.ptr,
                d_partial.data.ptr,
                cp.uint64(num_elements),
            )
        end = stream.record(options=event_opts)
        end.sync()

        time_ms = (end - start) / iterations
        bandwidth = (num_elements * 4) / (time_ms * 1e6)
        print(f"\nKernel time: {time_ms:.3f} ms, Bandwidth: {bandwidth:.1f} GB/s")

        return test1 and test2 and test3

    finally:
        # Explicit resource cleanup
        cp.cuda.Stream.null.use()
        stream.close()


def main() -> None:
    """Entry point."""
    success = run_sample()
    if success:
        print("\nDone")
    else:
        print("\nSome tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
