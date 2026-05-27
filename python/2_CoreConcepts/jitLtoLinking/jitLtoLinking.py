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
JIT Compilation and Link-Time Optimization with cuda.core

Real-world GPU code is rarely a single source string. Libraries ship a
"main" kernel that is compiled once, then link in user-supplied device
functions at runtime to customize behavior without recompiling the whole
program.

cuda.core exposes this pattern through ``Program`` (NVRTC compilation)
and ``Linker`` (JIT linking of multiple object codes). Two modes are
shown here:

  * **PTX linking**: compile each translation unit with
    ``relocatable_device_code=True`` to PTX and link to a CUBIN.
    The two modules remain independently compiled: no cross-module
    inlining.

  * **LTO (Link-Time Optimization)**: compile each translation unit
    with ``link_time_optimization=True`` to LTO IR, then link with
    ``LinkerOptions(link_time_optimization=True)``. The linker reruns
    the optimizer across both modules and can inline the device function
    into the main kernel, typically matching a single-source build.

The same kernel math runs in both modes and is verified against a
NumPy reference.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))

try:
    import cupy as cp
    import numpy as np
    from cuda.core import (
        Device,
        LaunchConfig,
        Linker,
        LinkerOptions,
        Program,
        ProgramOptions,
        launch,
    )
    from cuda_samples_utils import print_gpu_info  # noqa: E402
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


# --------------------------------------------------------------------------
# Module A: the "library" main kernel. It forwards each element through a
# user-supplied device function (resolved at link time) and writes the result.
# --------------------------------------------------------------------------
MAIN_SRC = r"""
// Forward declare the user-supplied hook. Its definition lives in a separate
// translation unit and is resolved by the Linker at runtime.
extern "C" __device__ float user_transform(float x);

extern "C" __global__
void apply_transform(const float* __restrict__ in,
                     float* __restrict__ out,
                     size_t N)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < N; i += stride) {
        out[i] = user_transform(in[i]);
    }
}
"""

# --------------------------------------------------------------------------
# Module B: the user-supplied "plug-in" device function. A different
# implementation of ``user_transform`` here produces different results without
# rebuilding MAIN_SRC.
# --------------------------------------------------------------------------
USER_SRC = r"""
extern "C" __device__
float user_transform(float x)
{
    // A deliberately non-trivial expression so LTO has something to inline /
    // optimize across the module boundary.
    float y = x * x + 3.0f * x - 1.0f;
    return y > 0.0f ? y : 0.0f;
}
"""


def host_reference(x: np.ndarray) -> np.ndarray:
    y = x * x + 3.0 * x - 1.0
    return np.where(y > 0.0, y, 0.0).astype(np.float32)


def link_ptx(device):
    """Compile both modules to PTX and link them into a cubin (no LTO)."""
    prog_opts = ProgramOptions(
        std="c++17", arch=f"sm_{device.arch}", relocatable_device_code=True
    )
    main_obj = Program(MAIN_SRC, "c++", options=prog_opts).compile("ptx")
    user_obj = Program(USER_SRC, "c++", options=prog_opts).compile("ptx")

    linker = Linker(main_obj, user_obj, options=LinkerOptions(arch=f"sm_{device.arch}"))
    return linker.link("cubin")


def link_lto(device):
    """Compile both modules to LTO IR and link with LTO enabled."""
    prog_opts = ProgramOptions(
        std="c++17", arch=f"sm_{device.arch}", link_time_optimization=True
    )
    main_obj = Program(MAIN_SRC, "c++", options=prog_opts).compile("ltoir")
    user_obj = Program(USER_SRC, "c++", options=prog_opts).compile("ltoir")

    linker_opts = LinkerOptions(
        arch=f"sm_{device.arch}", link_time_optimization=True
    )
    linker = Linker(main_obj, user_obj, options=linker_opts)
    return linker.link("cubin")


def run_one_mode(mode, module, stream, d_in, d_out, size, expected):
    kernel = module.get_kernel("apply_transform")
    config = LaunchConfig(grid=(size + 255) // 256, block=256)
    launch(
        stream,
        config,
        kernel,
        d_in.data.ptr,
        d_out.data.ptr,
        np.uint64(size),
    )
    stream.sync()
    actual = cp.asnumpy(d_out)
    if not np.allclose(actual, expected, rtol=1e-5, atol=1e-5):
        max_err = np.max(np.abs(actual - expected))
        print(f"  [{mode}] verification FAILED (max_err={max_err})")
        return False
    print(f"  [{mode}] result verified against NumPy reference")
    return True


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="JIT + LTO linking of two device modules with cuda.core"
    )
    parser.add_argument(
        "--elements", type=int, default=1 << 16,
        help="Number of float32 elements (default: 65536)",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    args = parser.parse_args()

    device = Device(args.device)
    device.set_current()
    print_gpu_info(device)

    stream = device.create_stream()
    cp.cuda.ExternalStream(int(stream.handle)).use()

    try:
        N = args.elements
        rng = np.random.default_rng(seed=0)
        host_in = rng.standard_normal(N).astype(np.float32)
        expected = host_reference(host_in)

        d_in = cp.asarray(host_in)
        d_out = cp.empty(N, dtype=cp.float32)
        device.sync()

        print("\n[1] PTX linking (no LTO)")
        ptx_module = link_ptx(device)
        ok_ptx = run_one_mode("ptx", ptx_module, stream, d_in, d_out, N, expected)

        d_out.fill(0)
        device.sync()

        print("\n[2] LTO linking (link-time optimization)")
        lto_module = link_lto(device)
        ok_lto = run_one_mode("lto", lto_module, stream, d_in, d_out, N, expected)

        print()
        if ok_ptx and ok_lto:
            print("Both PTX and LTO linked kernels produced matching results. Done")
            return 0
        return 1
    finally:
        stream.close()
        cp.cuda.Stream.null.use()


if __name__ == "__main__":
    sys.exit(main())
