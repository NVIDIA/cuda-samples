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
Vector Addition using CUDA Core API

This sample demonstrates element-wise vector addition: C = A + B
using cuda.core for runtime compilation and kernel launch.
"""

import sys
from pathlib import Path

# Add parent directory to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))
from cuda_samples_utils import verify_array_result  # noqa: E402

try:
    import cupy as cp
    from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


# CUDA kernel source code
VECTOR_ADD_KERNEL = """
/**
 * CUDA Kernel for vector addition
 * Computes the vector addition of A and B into C.
 */
template<typename T>
__global__ void vectorAdd(const T *A, const T *B, T *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}
"""


def vector_add_cuda_core(num_elements=50000, device_id=0, verify=True):
    """
    Perform vector addition using cuda.core API.

    Parameters
    ----------
    num_elements : int
        Number of elements in each vector
    device_id : int
        CUDA device ID to use
    verify : bool
        Whether to verify the result

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Initialize device
        print("[Vector addition using CUDA Core API]")
        device = Device(device_id)
        device.set_current()

        print(f"Device: {device.name}")
        print(f"Compute Capability: sm_{device.arch}")

        stream = device.create_stream()

        # Compile kernel
        print("Compiling kernel 'vectorAdd<float>'...")
        program_options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
        program = Program(VECTOR_ADD_KERNEL, code_type="c++", options=program_options)
        module = program.compile("cubin", name_expressions=("vectorAdd<float>",))
        kernel = module.get_kernel("vectorAdd<float>")
        print("Kernel compiled successfully")

        # Allocate and initialize vectors
        print(f"[Vector addition of {num_elements} elements]")
        dtype = cp.float32

        a = cp.random.rand(num_elements).astype(dtype)
        b = cp.random.rand(num_elements).astype(dtype)
        c = cp.empty(num_elements, dtype=dtype)

        # Synchronize before kernel launch
        device.sync()

        # Configure and launch kernel
        threads_per_block = 256
        blocks_per_grid = (num_elements + threads_per_block - 1) // threads_per_block

        print(
            f"CUDA kernel launch with {blocks_per_grid} blocks "
            f"of {threads_per_block} threads"
        )

        config = LaunchConfig(grid=blocks_per_grid, block=threads_per_block)

        # Launch kernel
        launch(
            stream,
            config,
            kernel,
            a.data.ptr,
            b.data.ptr,
            c.data.ptr,
            cp.int32(num_elements),
        )
        stream.sync()

        # Verify result
        if verify:
            print("Verifying result...")
            expected = a + b
            if not verify_array_result(c, expected):
                return False

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """
    Main entry point for the vector addition sample.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Vector Addition using CUDA Core API")
    parser.add_argument(
        "--elements",
        type=int,
        default=50000,
        help="Number of elements in vectors (default: 50000)",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device ID (default: 0)"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip result verification"
    )

    args = parser.parse_args()

    if args.elements <= 0:
        print("Error: Number of elements must be positive")
        return 1

    success = vector_add_cuda_core(
        num_elements=args.elements, device_id=args.device, verify=not args.no_verify
    )

    if success:
        print("\nDone")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
