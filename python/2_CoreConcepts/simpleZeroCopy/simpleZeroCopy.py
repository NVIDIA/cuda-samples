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

import argparse
import ctypes
import sys
from pathlib import Path

try:
    import numpy as np
    from cuda.bindings import runtime as cuda_rt
    from cuda.core import (
        Device,
        LaunchConfig,
        Program,
        ProgramOptions,
        launch,
    )
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Add parent directory to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))


def _mapped_host_alloc(num_floats, stream):
    """
    Allocate page-locked host memory mapped for device access; return
    (host_ptr, device_ptr) for CPU views and for ``launch()``.
    """
    nbytes = int(num_floats) * np.dtype(np.float32).itemsize
    if nbytes <= 0:
        return 0, 0
    err, h_ptr = cuda_rt.cudaHostAlloc(
        nbytes, cuda_rt.cudaHostAllocMapped | cuda_rt.cudaHostAllocPortable
    )
    if err != cuda_rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaHostAlloc failed: {err}")
    err, d_ptr = cuda_rt.cudaHostGetDevicePointer(h_ptr, 0)
    if err != cuda_rt.cudaError_t.cudaSuccess:
        cuda_rt.cudaFreeHost(h_ptr)
        raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
    # Ensure prior work on this stream is visible before host fills buffers.
    if stream is not None:
        stream.sync()
    return h_ptr, d_ptr


def _float_view(host_ptr, num_floats):
    return np.frombuffer(
        (ctypes.c_float * num_floats).from_address(host_ptr),
        dtype=np.float32,
        count=num_floats,
    )


# CUDA C++: vector add with grid-stride loop
VECTOR_ADD_KERNEL = """
extern "C" __global__
void vectorAddGPU(float* c, const float* a, const float* b, int N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < N; i += stride) {
        c[i] = a[i] + b[i];
    }
}
"""


def run(num_elements=1048576):
    """
    Zero-copy vector add: map host memory, launch kernel with device
    pointers, validate on CPU.

    This function shows how to:
    1. Allocate pinned (page-locked) host memory
    2. Map host memory into GPU address space (zero-copy)
    3. Access host memory directly from GPU kernel
    4. Validate results

    Parameters
    ----------
    num_elements : int
        Number of elements in vectors (default: 1048576)
    """
    print("\n" + "=" * 70)
    print("simpleZeroCopy - CUDA Python Sample")
    print("=" * 70)

    # Initialize device
    device = Device()
    device.set_current()
    major, minor = device.compute_capability

    print("\nDevice Information:")
    print(f"  Name: {device.name}")
    print(f"  Compute Capability: {major}.{minor}")

    # Create stream
    stream = device.create_stream()
    mapped_host_ptrs = []

    try:
        print(
            "\n> Memory: mapped pinned host "
            "(cudaHostAlloc + cudaHostGetDevicePointer)"
        )

        print("\nCompiling CUDA kernel...")
        program_options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
        prog = Program(VECTOR_ADD_KERNEL, code_type="c++", options=program_options)
        mod = prog.compile("cubin")
        kernel = mod.get_kernel("vectorAddGPU")
        print("  Kernel compiled successfully")

        bytes_total = num_elements * np.dtype(np.float32).itemsize
        print("\nAllocating memory:")
        print(f"  Vector size: {num_elements:,} elements")
        print(f"  Memory per vector: {bytes_total / (1024**2):.2f} MB")
        print(f"  Total memory: {3 * bytes_total / (1024**2):.2f} MB")

        print("\n> Allocating mapped pinned host memory...")
        h_a, d_a = _mapped_host_alloc(num_elements, stream)
        mapped_host_ptrs.append(h_a)
        h_b, d_b = _mapped_host_alloc(num_elements, stream)
        mapped_host_ptrs.append(h_b)
        h_c, d_c = _mapped_host_alloc(num_elements, stream)
        mapped_host_ptrs.append(h_c)

        a = _float_view(h_a, num_elements)
        b = _float_view(h_b, num_elements)
        c = _float_view(h_c, num_elements)

        print("  Mapped host memory allocated successfully")

        print("\n> Initializing vectors on host...")
        rng = np.random.default_rng(42)
        a[:] = rng.random(num_elements).astype(np.float32)
        b[:] = rng.random(num_elements).astype(np.float32)
        c[:] = 0

        print("> Computing reference result on CPU...")
        reference = a + b

        print("\n> Launching vectorAddGPU kernel...")
        print("  Note: GPU accesses host memory directly (zero-copy)")

        block_size = 256
        grid_size = (num_elements + block_size - 1) // block_size
        config = LaunchConfig(grid=grid_size, block=block_size)

        # Pass device pointers from cudaHostGetDevicePointer, not raw host VAs.
        launch(
            stream,
            config,
            kernel,
            int(d_c),
            int(d_a),
            int(d_b),
            np.int32(num_elements),
        )
        stream.sync()

        print("  Kernel execution complete")

        print("\n> Checking results from vectorAddGPU()...")
        print(f"  Comparing {num_elements:,} elements...")

        # ``c`` is a host view of the same buffer; no cudaMemcpy D2H needed.
        if np.allclose(c, reference, rtol=1e-5, atol=1e-6):
            error_norm = np.linalg.norm(c - reference)
            ref_norm = np.linalg.norm(reference)
            relative_error = error_norm / ref_norm
            print(f"  Relative error: {relative_error:.6e}")
            print("  Validation PASSED")
            success = True
        else:
            max_error = np.max(np.abs(c - reference))
            print(f"  Max error: {max_error}")
            print("  Validation FAILED")
            success = False

        print("\n" + "=" * 70)
        if success:
            print("simpleZeroCopy completed successfully!")
        else:
            print("simpleZeroCopy FAILED!")
        print("=" * 70 + "\n")

        return 0 if success else 1
    finally:
        for h in reversed(mapped_host_ptrs):
            if h:
                cuda_rt.cudaFreeHost(h)
        stream.close()


def main():
    """Parse CLI, call ``run()``, and exit with validation status."""
    parser = argparse.ArgumentParser(
        description="Demonstrate zero-copy memory access with CUDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simpleZeroCopy.py
  python simpleZeroCopy.py --num_elements 2097152
What is Zero-Copy Memory?
  Zero-copy allows the GPU to directly access host (CPU) memory without
  explicit memory transfers. This is useful for:
  - Small data that doesn't benefit from explicit transfers
  - Data that is accessed infrequently
  - Integrated GPUs that share memory with CPU

  Trade-offs:
  - Slower than device memory (PCIe bandwidth limited)
  - No explicit transfers needed (simpler code)
  - Good for discrete GPUs with small data
  - Excellent for integrated GPUs (e.g., Tegra)
        """,
    )

    parser.add_argument(
        "--num_elements",
        type=int,
        default=1048576,
        help="Number of elements in vectors (default: 1048576)",
    )

    args = parser.parse_args()

    if args.num_elements <= 0:
        print("Error: num_elements must be positive")
        sys.exit(1)

    try:
        exit_code = run(num_elements=args.num_elements)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
