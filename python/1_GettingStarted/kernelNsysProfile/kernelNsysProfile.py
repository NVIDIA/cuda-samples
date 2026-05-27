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
Kernel Nsys Profiling Sample - CUDA C++ Kernel Profiling with cuda.core

This sample demonstrates how to profile custom CUDA C++ kernels compiled and
launched with cuda.core using NVIDIA Nsight Systems.

The sample implements three common GPU operations as custom CUDA C++ kernels:
- Vector addition: c = a + b
- SAXPY: y = alpha * x + y
- Vector transform: sqrt(x*x + 1) + sin(x)

Use Nsight Systems to analyze:
- Custom kernel execution times
- Kernel launch patterns and overhead
- GPU utilization and memory access patterns
- NVTX markers for structured profiling

Workflow:
- Phase 1: Create GPU arrays
- Phase 2: Compile and execute cuda.core custom kernels (profiling focus)
- Phase 3: Verify correctness with CuPy reference implementation
- Phase 4: Validate results
"""

import argparse
import sys
from pathlib import Path

try:
    import cupy as cp
    import numpy as np
    import nvtx
    from cuda.core import Device, LaunchConfig, launch
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Add parent directory to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))
from cuda_samples_utils import verify_array_result  # noqa: E402

# CUDA C++ kernel definitions
# For larger projects, separating kernels into a separate file is also valid.
KERNELS_CODE = """
template<typename T>
__global__ void vector_add(const T* a, const T* b, T* c, size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = tid; i < N; i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

template<typename T>
__global__ void saxpy(const T alpha, const T* x, T* y, size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = tid; i < N; i += gridDim.x * blockDim.x) {
        y[i] = alpha * x[i] + y[i];
    }
}

template<typename T>
__global__ void vector_transform(const T* a, T* b, size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = tid; i < N; i += gridDim.x * blockDim.x) {
        T val = a[i];
        b[i] = sqrt(val * val + T(1.0)) + sin(val);
    }
}
"""


def get_cuda_core_kernels(device):
    """
    Compile cuda.core kernels and return them.

    Args:
        device: cuda.core.Device object

    Returns:
        dict: Dictionary of compiled kernels
    """
    from cuda.core import Program, ProgramOptions

    # Compile all kernels at once
    program_options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
    prog = Program(KERNELS_CODE, code_type="c++", options=program_options)
    mod = prog.compile(
        "cubin",
        name_expressions=(
            "vector_add<float>",
            "saxpy<float>",
            "vector_transform<float>",
        ),
    )

    # Extract individual kernels
    return {
        "vector_add": mod.get_kernel("vector_add<float>"),
        "saxpy": mod.get_kernel("saxpy<float>"),
        "vector_transform": mod.get_kernel("vector_transform<float>"),
    }


def run(size):
    """Main execution function"""

    # =================================================================
    # Device Initialization using cuda.core
    # =================================================================
    with nvtx.annotate("Device Initialization", color="green"):
        try:
            # Create device object (defaults to device 0)
            dev = Device()
            dev.set_current()

            print()
            print(f"Device: {dev.name}")
            print(f"Compute Capability: sm_{dev.arch}")
            print()

            # Synchronize device
            dev.sync()

        except Exception as e:
            print("ERROR: CUDA initialization failed!")
            print(f"Error: {e}")
            sys.exit(1)

    print("Profiling cuda.core Custom Kernels")
    print(f"Array size: {size:,}\n")

    # Constant for SAXPY operation
    alpha = 2.5

    # Initialize random seed
    rng = cp.random.default_rng(42)

    # =================================================================
    # Phase 1: Create GPU Arrays with CuPy
    # =================================================================
    with nvtx.annotate("Create GPU Arrays", color="yellow"):
        a_gpu = rng.standard_normal(size, dtype=cp.float32)
        b_gpu = rng.standard_normal(size, dtype=cp.float32)
        dev.sync()

        print("Phase 1: Created arrays on GPU with CuPy")
        print(f"  Array shape: {a_gpu.shape}")
        print(f"  Array dtype: {a_gpu.dtype}")
        print(
            f"  Array a - Mean: {float(cp.mean(a_gpu)):.4f}, "
            f"Std: {float(cp.std(a_gpu)):.4f}"
        )
        print(
            f"  Array b - Mean: {float(cp.mean(b_gpu)):.4f}, "
            f"Std: {float(cp.std(b_gpu)):.4f}\n"
        )

    # =================================================================
    # Phase 2: cuda.core Custom Kernels on GPU
    # =================================================================
    with nvtx.annotate("cuda.core Custom Kernels", color="purple"):
        print("Phase 2: cuda.core custom CUDA C++ kernels on GPU")

        # Create a stream for cuda.core operations
        stream = dev.create_stream()
        try:
            with nvtx.annotate("Compile Kernels", color="cyan"):
                kernels_dict = get_cuda_core_kernels(dev)
                stream.sync()
                print("Compiled custom CUDA C++ kernels")

            # Prepare launch configuration
            # Grid-stride loops in kernels handle any grid size robustly
            block = 256
            grid = (size + block - 1) // block
            config = LaunchConfig(grid=grid, block=block)

            # Execute cuda.core vector_add kernel
            with nvtx.annotate("Vector Add (cuda.core)", color="cyan"):
                c_cuda = cp.empty_like(a_gpu)
                launch(
                    stream,
                    config,
                    kernels_dict["vector_add"],
                    a_gpu.data.ptr,
                    b_gpu.data.ptr,
                    c_cuda.data.ptr,
                    cp.uint64(size),
                )
                stream.sync()

            # Execute cuda.core SAXPY kernel
            with nvtx.annotate("SAXPY (cuda.core)", color="cyan"):
                y_cuda = b_gpu.copy()
                launch(
                    stream,
                    config,
                    kernels_dict["saxpy"],
                    np.float32(alpha),
                    a_gpu.data.ptr,
                    y_cuda.data.ptr,
                    cp.uint64(size),
                )
                stream.sync()

            # Execute cuda.core vector_transform kernel
            with nvtx.annotate("Vector Transform (cuda.core)", color="cyan"):
                transform_cuda = cp.empty_like(a_gpu)
                launch(
                    stream,
                    config,
                    kernels_dict["vector_transform"],
                    a_gpu.data.ptr,
                    transform_cuda.data.ptr,
                    cp.uint64(size),
                )
                stream.sync()

            print("Vector Addition (custom kernel)")
            print("SAXPY (custom kernel)")
            print("Vector Transform (custom kernel)\n")
        finally:
            stream.close()

    # =================================================================
    # Phase 3: Generate Reference Results with CuPy (for verification)
    # =================================================================
    with nvtx.annotate("Generate Reference Results", color="blue"):
        print("Phase 3: Generate reference results for verification")

        with nvtx.annotate("Vector Add (Reference)", color="cyan"):
            c_cupy = a_gpu + b_gpu
            dev.sync()

        with nvtx.annotate("SAXPY (Reference)", color="cyan"):
            y_cupy = alpha * a_gpu + b_gpu
            dev.sync()

        with nvtx.annotate("Vector Transform (Reference)", color="cyan"):
            transform_cupy = cp.sqrt(a_gpu * a_gpu + 1.0) + cp.sin(a_gpu)
            dev.sync()

        print("Reference results generated\n")

    # =================================================================
    # Phase 4: Verify Kernel Correctness
    # =================================================================
    with nvtx.annotate("Verification", color="green"):
        print("Phase 4: Verify kernel correctness")

        # Verify custom kernels against reference results
        # Use relaxed tolerances for single-precision float comparisons
        # Small differences can occur due to instruction ordering and
        # compiler optimizations
        print("  Validating cuda.core kernels:")

        print("    Vector Add: ", end="")
        vec_add_match = verify_array_result(c_cuda, c_cupy, rtol=1e-5, atol=1e-6)

        print("    SAXPY:      ", end="")
        saxpy_match = verify_array_result(y_cuda, y_cupy, rtol=1e-5, atol=1e-6)

        print("    Transform:  ", end="")
        transform_match = verify_array_result(
            transform_cuda, transform_cupy, rtol=1e-5, atol=1e-6
        )

        all_pass = vec_add_match and saxpy_match and transform_match

        if not all_pass:
            print("\n  ERROR: Kernel verification failed!")
            return 1
        print()

    # Final synchronization
    dev.sync()
    print("The sample is complete PASSED!")


def main():
    parser = argparse.ArgumentParser(
        description="Kernel Nsys Profiling - Profile custom CUDA C++ "
        "kernels with cuda.core"
    )
    parser.add_argument(
        "-n",
        "--array-size",
        type=int,
        default=50000,
        metavar="N",
        help="Array size (default: 50,000)",
    )

    args = parser.parse_args()
    run(size=args.array_size)


if __name__ == "__main__":
    main()
