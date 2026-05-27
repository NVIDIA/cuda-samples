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
Simple Print - Printing from CUDA Kernels

This sample demonstrates how to print output from CUDA kernels using printf().
It shows:
1. Device management with cuda.core.Device
2. Compiling CUDA C++ code that uses printf()
3. Launching kernels with 2D grids and 3D blocks
4. Seeing kernel output printed to stdout
5. Using Numba CUDA for Pythonic kernel authoring

This sample demonstrates both approaches:
- CUDA C++ kernels compiled via cuda.core.Program (more control, C++ features)
- Numba CUDA kernels (more Pythonic, easier to write)

This is the Python equivalent of the C++ simplePrintf sample.
"""

import sys
import traceback

try:
    from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

try:
    from numba import cuda as numba_cuda

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not found. Numba CUDA example will be skipped.")
    print("To install: pip install numba")


# CUDA C++ kernel with printf
# This kernel prints the block index, thread index, and a value from each thread
PRINTF_KERNEL = """
extern "C"
__global__ void printKernel(int val) {
    // Calculate linear block index from 2D grid
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;

    // Calculate linear thread index from 3D block
    int threadId = threadIdx.z * blockDim.x * blockDim.y +
                   threadIdx.y * blockDim.x +
                   threadIdx.x;

    // Print from each thread
    printf("[%d, %d]:\\t\\tValue is: %d\\n", blockId, threadId, val);
}
"""


# Numba CUDA kernel - Pythonic equivalent using numba.cuda.grid()
# This demonstrates the same functionality using Numba's Python-based kernel syntax
if NUMBA_AVAILABLE:

    @numba_cuda.jit
    def numba_print_kernel(val):
        """
        Numba CUDA kernel showing the *recommended* grid() indexing style,
        while also relating it to the classic CUDA C++ blockId/threadId.

        - Primary view: global 3D coordinates from numba.cuda.grid(3)
          (modern, Pythonic way to index work for a 3D thread layout).
        - Secondary view: linear blockId / threadId matching the CUDA C++
          printf sample, to help CUDA C++ users connect the two models.
        """
        # Modern / recommended view: global 3D thread coordinates
        x, y, z = numba_cuda.grid(3)

        # Classic CUDA-style indices, same formulas as the C++ sample
        block_id = numba_cuda.blockIdx.y * numba_cuda.gridDim.x + numba_cuda.blockIdx.x

        thread_id = (
            numba_cuda.threadIdx.z * numba_cuda.blockDim.x * numba_cuda.blockDim.y
            + numba_cuda.threadIdx.y * numba_cuda.blockDim.x
            + numba_cuda.threadIdx.x
        )

        # Print both views side-by-side
        # Note: Numba print() adds spaces between comma-separated args
        print(
            "global[",
            x,
            ",",
            y,
            ",",
            z,
            "] -> [",
            block_id,
            ",",
            thread_id,
            "]:\t\tValue is:",
            val,
        )


def run_cuda_cpp_kernel(device, test_value=10):
    """
    Demonstrate printing from CUDA C++ kernel compiled with cuda.core.

    This approach gives you full access to CUDA C++ features and allows
    for more complex kernel implementations.
    """
    print("=" * 70)
    print("METHOD 1: CUDA C++ Kernel (via cuda.core.Program)")
    print("=" * 70)
    print("Advantage: Full C++ features, better for complex kernels")
    print()

    # Compile the kernel
    print("Compiling CUDA C++ kernel...")
    program_options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
    prog = Program(PRINTF_KERNEL, code_type="c++", options=program_options)
    mod = prog.compile("cubin", name_expressions=("printKernel",))
    kernel = mod.get_kernel("printKernel")
    print("Kernel compiled successfully.\n")

    # Create stream for kernel execution
    stream = device.create_stream()

    # Configure kernel launch
    # Using 2D grid (2x2) and 3D blocks (2x2x2)
    grid_x, grid_y = 2, 2
    block_x, block_y, block_z = 2, 2, 2

    print("Kernel configuration:")
    print(f"  Grid:  ({grid_x}, {grid_y})")
    print(f"  Block: ({block_x}, {block_y}, {block_z})")
    print(f"  Total threads: {grid_x * grid_y * block_x * block_y * block_z}")
    print()

    # Launch configuration with 2D grid and 3D block
    config = LaunchConfig(grid=(grid_x, grid_y), block=(block_x, block_y, block_z))

    print(f"Launching kernel with value={test_value}. Output:\n")
    try:
        # Launch kernel
        launch(stream, config, kernel, test_value)

        # Synchronize to ensure printf output is flushed
        stream.sync()

        print("\nCUDA C++ kernel execution complete.")
    except Exception as e:
        print(f"\nError during kernel execution: {e}")
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        stream.close()

    return 0


def run_numba_kernel(device, test_value=10):
    """
    Demonstrate printing from a Numba CUDA kernel.

    This example uses numba.cuda.grid(3) as the primary indexing mechanism
    (recommended modern style), and also prints the equivalent blockId /
    threadId used in the CUDA C++ printf sample for side-by-side comparison.

    Uses cuda.core APIs for stream management, demonstrating interoperability
    between Numba CUDA kernels and cuda.core infrastructure.
    """
    print("\n")
    print("=" * 70)
    print("METHOD 2: Numba CUDA Kernel (Pythonic / modern indexing)")
    print("=" * 70)
    print("Advantage: Uses numba.cuda.grid(3) for global indexing,")
    print("           while still showing classic CUDA C++ IDs for reference.")
    print("           Uses cuda.core for stream management (interoperability).")
    print()

    # Same launch configuration as the C++ version
    grid_x, grid_y = 2, 2
    block_x, block_y, block_z = 2, 2, 2

    print("Kernel configuration:")
    print(f"  Grid:  ({grid_x}, {grid_y})")
    print(f"  Block: ({block_x}, {block_y}, {block_z})")
    print(f"  Total threads: {grid_x * grid_y * block_x * block_y * block_z}")
    print()

    # Use cuda.core stream (same as C++ example) instead of numba.cuda.stream()
    stream = device.create_stream()

    print(f"Launching Numba kernel (grid(3) + classic IDs) with value={test_value}:")
    print("Uses numba.cuda.grid(3) to get global (x, y, z),")
    print("and prints the corresponding blockId/threadId like the C++ sample.")
    print("Stream managed by cuda.core for consistency with C++ example.\n")

    try:
        # Launch Numba kernel on cuda.core stream
        numba_print_kernel[(grid_x, grid_y), (block_x, block_y, block_z), stream](
            test_value
        )

        # Synchronize cuda.core stream (same as C++ example)
        stream.sync()
        print("\nNumba CUDA kernel execution complete.")
    except Exception as e:
        print(f"\nError during Numba kernel execution: {e}")
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        stream.close()

    return 0


def main():
    """Main function demonstrating printing from CUDA kernels using both approaches"""

    print("Simple Print - Printing from CUDA Kernels")
    print("Demonstrating both CUDA C++ and Numba CUDA approaches")
    print()
    # Initialize device
    device = Device()
    device.set_current()

    # Get device properties
    print(f"Device: {device.name}")
    print(f"Compute Capability: sm_{device.arch}")
    print()

    # Value to pass to both kernels
    test_value = 10

    # Run CUDA C++ kernel
    result = run_cuda_cpp_kernel(device, test_value)
    if result != 0:
        return result

    # Run Numba kernel if available
    if NUMBA_AVAILABLE:
        result = run_numba_kernel(device, test_value)
        if result != 0:
            return result
    else:
        print("\n" + "=" * 70)
        print("Numba CUDA example skipped (numba not installed)")
        print("To run the Numba example: pip install numba")
        print("=" * 70)

    print("\n" + "=" * 70)
    print("Done! Both kernel approaches demonstrated successfully.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
