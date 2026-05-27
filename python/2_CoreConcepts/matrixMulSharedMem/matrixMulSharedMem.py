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
Matrix Multiplication with Shared Memory (GEMM)

Demonstrates efficient matrix multiplication using:
- nvmath.linalg.advanced.Matmul for high-performance GEMM via cuBLASLt
- Custom CUDA kernel with tiling, shared memory, and loop unrolling

Uses cuda.core APIs with CuPy arrays via ExternalStream.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))

try:
    import cupy as cp
    import numpy as np
    import nvmath.linalg.advanced as nvmath_advanced
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


TILE_SIZE: int = 16

MATMUL_KERNEL: str = r"""
#define TILE_SIZE 16

extern "C" __global__
void matmul_shared(const float* A, const float* B, float* C,
                   int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;

        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += 4) {
            sum += As[ty][k]     * Bs[k][tx];
            sum += As[ty][k + 1] * Bs[k + 1][tx];
            sum += As[ty][k + 2] * Bs[k + 2][tx];
            sum += As[ty][k + 3] * Bs[k + 3][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"""


def run_matmul_benchmark(
    m: int = 1024,
    n: int = 1024,
    k: int = 1024,
    device_id: int = 0,
    num_iterations: int = 10,
) -> bool:
    """Run matrix multiplication benchmark comparing nvmath vs custom kernel."""
    print("=" * 60)
    print("Matrix Multiplication with Shared Memory (GEMM)")
    print("=" * 60)

    # Initialize device and stream
    device = Device(device_id)
    device.set_current()
    stream = device.create_stream()
    print(f"\nDevice: {device.name}")
    print(f"Compute Capability: sm_{device.arch}")

    # Make CuPy use our cuda.core stream
    cp.cuda.ExternalStream(int(stream.handle)).use()

    # Compile custom kernel
    arch = f"sm_{device.arch}"
    program = Program(MATMUL_KERNEL, code_type="c++", options=ProgramOptions(arch=arch))
    kernel = program.compile(target_type="cubin").get_kernel("matmul_shared")
    print("Custom kernel compiled ✓")

    # Setup
    print(f"\nMatrix: A({m}x{k}) × B({k}x{n}) = C({m}x{n})")
    total_ops = 2 * m * n * k
    event_opts = EventOptions(enable_timing=True)

    # Allocate matrices
    rng = cp.random.default_rng(42)
    d_A = rng.random((m, k), dtype=cp.float32)
    d_B = rng.random((k, n), dtype=cp.float32)
    d_C_custom = cp.zeros((m, n), dtype=cp.float32)

    success = True
    try:
        # -------------------------------------------------------------------------
        # nvmath GEMM (cuBLASLt)
        # -------------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("NVMATH (cuBLASLt) - plan once, execute many")
        print("-" * 60)

        with nvmath_advanced.Matmul(d_A, d_B, stream=int(stream.handle)) as mm:
            mm.plan()
            d_C_nvmath = mm.execute()
            stream.sync()

            start = stream.record(options=event_opts)
            for _ in range(num_iterations):
                d_C_nvmath = mm.execute()
            end = stream.record(options=event_opts)
            end.sync()

        nvmath_ms = (end - start) / num_iterations
        nvmath_gflops = (total_ops / 1e9) / (nvmath_ms / 1e3)
        print(f"Time: {nvmath_ms:.3f} ms | {nvmath_gflops:.2f} GFLOPS")

        # -------------------------------------------------------------------------
        # Custom kernel (tiled + shared memory + unroll)
        # -------------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("CUSTOM KERNEL (tiled + shared memory + unroll)")
        print("-" * 60)

        block = (TILE_SIZE, TILE_SIZE)
        grid = ((n + TILE_SIZE - 1) // TILE_SIZE, (m + TILE_SIZE - 1) // TILE_SIZE)
        config = LaunchConfig(grid=grid, block=block)

        launch(
            stream,
            config,
            kernel,
            d_A.data.ptr,
            d_B.data.ptr,
            d_C_custom.data.ptr,
            np.int32(m),
            np.int32(n),
            np.int32(k),
        )
        stream.sync()

        start = stream.record(options=event_opts)
        for _ in range(num_iterations):
            launch(
                stream,
                config,
                kernel,
                d_A.data.ptr,
                d_B.data.ptr,
                d_C_custom.data.ptr,
                np.int32(m),
                np.int32(n),
                np.int32(k),
            )
        end = stream.record(options=event_opts)
        end.sync()

        custom_ms = (end - start) / num_iterations
        custom_gflops = (total_ops / 1e9) / (custom_ms / 1e3)
        print(f"Time: {custom_ms:.3f} ms | {custom_gflops:.2f} GFLOPS")

        # -------------------------------------------------------------------------
        # Verification
        # -------------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("VERIFICATION")
        print("-" * 60)

        d_C_ref = d_A @ d_B

        # Host-side verification: cp.allclose triggers NVRTC failure on sm_120
        # (ldexp_cexp undefined). Use asnumpy + np.allclose instead.
        ref_host = cp.asnumpy(d_C_ref)
        for name, d_C in [("nvmath", d_C_nvmath), ("custom", d_C_custom)]:
            print(f"{name}: ", end="")
            passed = np.allclose(cp.asnumpy(d_C), ref_host, rtol=1e-4, atol=1e-4)
            print("Test PASSED" if passed else "Test FAILED")
            success = success and passed

        return success
    finally:
        cp.cuda.Stream.null.use()
        stream.close()


def main() -> bool:
    """Entry point. Returns True if benchmark passed."""
    return run_matmul_benchmark()


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
