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
TMA Tensor Map with cuda.core

This sample demonstrates how to use Tensor Memory Accelerator (TMA)
descriptors with cuda.core on Hopper and later GPUs (compute
capability >= 9.0). TMA enables efficient bulk data movement between
global and shared memory using hardware-managed tensor map
descriptors.

The sample:

  1. Creates a TMA tiled descriptor from a CuPy device array via
     ``StridedMemoryView.from_any_interface(...).as_tensor_map(...)``.
  2. Passes the descriptor by value (as ``__grid_constant__``) to a
     kernel that uses libcudacxx TMA/barrier wrappers to bulk-load a
     tile into shared memory.
  3. Reuses the same descriptor against a new source tensor with
     ``replace_address()`` to avoid rebuilding it.

On GPUs older than Hopper (sm < 90), the sample prints a diagnostic
and exits cleanly.

Ported from ``cuda_core/examples/tma_tensor_map.py`` in the
`cuda-python` repository.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))

try:
    import cupy as cp
    import numpy as np
    from cuda.core import (
        Device,
        LaunchConfig,
        Program,
        ProgramOptions,
        StridedMemoryView,
        launch,
    )
    from cuda.pathfinder import find_nvidia_header_directory, get_cuda_path_or_home
    from cuda_samples_utils import print_gpu_info
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


TILE_SIZE = 128  # elements per tile, must match the kernel constant

KERNEL_SRC = r"""
#include <cuda/barrier>

// Minimal definition of the 128-byte opaque tensor-map struct.
struct __align__(64) TensorMap { unsigned long long opaque[16]; };

static constexpr int TILE_SIZE = 128;
using TmaBarrier = cuda::barrier<cuda::thread_scope_block>;

extern "C"
__global__ void tma_copy(
    const __grid_constant__ TensorMap tensor_map,
    float* output,
    int N)
{
    __shared__ __align__(128) float smem[TILE_SIZE];
    __shared__ TmaBarrier bar;

    const int tid        = threadIdx.x;
    const int tile_start = blockIdx.x * TILE_SIZE;

    if (tid == 0)
    {
        init(&bar, 1);
    }
    __syncthreads();

    if (tid == 0)
    {
        cuda::device::experimental::cp_async_bulk_tensor_1d_global_to_shared(
            smem,
            reinterpret_cast<const CUtensorMap*>(&tensor_map),
            tile_start,
            bar);
        bar.wait(cuda::device::barrier_arrive_tx(bar, 1, TILE_SIZE * sizeof(float)));
    }
    __syncthreads();

    if (tid < TILE_SIZE)
    {
        const int idx = tile_start + tid;
        if (idx < N)
            output[idx] = smem[tid];
    }
}
"""


def _get_cccl_include_paths() -> list:
    """Locate the CUDA toolkit and libcudacxx (cccl) include directories.

    ``cuda.pathfinder.find_nvidia_header_directory`` searches pip-installed
    CUDA packages, conda environments, and the standard system install
    locations, so this works without requiring ``CUDA_PATH`` or
    ``CUDA_HOME``. ``get_cuda_path_or_home`` is used as a final fallback.
    """
    include_path: list = []

    # libcudacxx (cccl) - preferred, provides <cuda/barrier> used below.
    try:
        cccl_dir = find_nvidia_header_directory("cccl")
        if cccl_dir and os.path.isdir(cccl_dir):
            include_path.append(cccl_dir)
    except Exception:  # noqa: S110 - fallback probes continue below
        pass

    # CUDA runtime headers - needed for the CUtensorMap driver type.
    try:
        cudart_dir = find_nvidia_header_directory("cudart")
        if (
            cudart_dir
            and os.path.isdir(cudart_dir)
            and cudart_dir not in include_path
        ):
            include_path.append(cudart_dir)
    except Exception:  # noqa: S110 - fallback probes continue below
        pass

    # Fallback: use CUDA_PATH / CUDA_HOME when pathfinder comes up empty.
    if not include_path:
        cuda_path = get_cuda_path_or_home()
        if cuda_path is not None:
            cuda_include = os.path.join(cuda_path, "include")
            if os.path.isdir(cuda_include):
                include_path.append(cuda_include)
                cccl_include = os.path.join(cuda_include, "cccl")
                if os.path.isdir(cccl_include):
                    include_path.insert(0, cccl_include)

    if not include_path:
        print(
            "Could not locate CUDA toolkit headers.\n"
            "Tried cuda.pathfinder (pip/conda/system installs) and "
            "CUDA_PATH / CUDA_HOME; none succeeded.\n"
            "Set CUDA_HOME to your toolkit root (containing include/cccl "
            "and include/cuda_runtime.h) and retry.",
            file=sys.stderr,
        )
        sys.exit(1)
    return include_path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Use a TMA tensor map to bulk-copy data on Hopper+ GPUs"
    )
    parser.add_argument(
        "--elements",
        type=int,
        default=1024,
        help="Total number of float32 elements (must be a multiple of 128)",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    args = parser.parse_args()

    if args.elements % TILE_SIZE != 0:
        print(f"--elements must be a multiple of TILE_SIZE={TILE_SIZE}")
        return 1

    dev = Device(args.device)
    print_gpu_info(dev)

    arch = dev.compute_capability
    if arch < (9, 0):
        print(
            f"\nTMA requires compute capability >= 9.0 (Hopper or later); "
            f"this device is {arch.major}.{arch.minor}. Exiting cleanly."
        )
        return 0

    dev.set_current()
    include_path = _get_cccl_include_paths()

    # Compile with the CUBIN code type to target the exact device arch.
    prog = Program(
        KERNEL_SRC,
        code_type="c++",
        options=ProgramOptions(
            std="c++17",
            arch=f"sm_{dev.arch}",
            include_path=include_path,
        ),
    )
    mod = prog.compile("cubin")
    kernel = mod.get_kernel("tma_copy")

    # (1) Prepare input data and verify the initial TMA copy.
    n = args.elements
    src = cp.arange(n, dtype=cp.float32)
    output = cp.zeros(n, dtype=cp.float32)
    dev.sync()  # CuPy uses its own stream

    tensor_map = StridedMemoryView.from_any_interface(
        src, stream_ptr=-1
    ).as_tensor_map(box_dim=(TILE_SIZE,))

    n_tiles = n // TILE_SIZE
    config = LaunchConfig(grid=n_tiles, block=TILE_SIZE)
    launch(
        dev.default_stream,
        config,
        kernel,
        tensor_map,
        output.data.ptr,
        np.int32(n),
    )
    dev.sync()

    if not cp.array_equal(output, src):
        print("TMA copy produced incorrect results")
        return 1
    print(f"TMA copy verified: {n} elements across {n_tiles} tiles")

    # (2) Demonstrate replace_address() without rebuilding the descriptor.
    replacement = cp.full(n, fill_value=42.0, dtype=cp.float32)
    dev.sync()

    tensor_map.replace_address(replacement)

    output2 = cp.zeros(n, dtype=cp.float32)
    launch(
        dev.default_stream,
        config,
        kernel,
        tensor_map,
        output2.data.ptr,
        np.int32(n),
    )
    dev.sync()

    if not cp.array_equal(output2, replacement):
        print("replace_address produced incorrect results")
        return 1
    print("replace_address verified: descriptor reused with new source tensor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
