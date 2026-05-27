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
CUDA Graphs with cuda.core

CUDA graphs let you record a DAG of operations once, then replay the entire
graph with a single driver call. For workflows that issue many small kernels
this can significantly reduce CPU-side launch overhead.

This sample runs a three-stage elementwise pipeline (add -> multiply ->
subtract) in two modes:

  1. Individually launched kernels on a stream.
  2. A single CUDA graph that captures the same three launches and is
     replayed with ``graph.launch(stream)``.

We then measure the wall-clock time of each mode across many iterations to
illustrate the graph replay advantage for short kernels, and demonstrate that
a graph can be relaunched against new data (the pointers are baked in, but
the contents of those buffers are not).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))

try:
    import cupy as cp
    import numpy as np
    from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch
    from cuda_samples_utils import print_gpu_info  # noqa: E402
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


PIPELINE_KERNELS = r"""
extern "C" __global__
void vec_add(const float* A, const float* B, float* C, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < N; i += stride) C[i] = A[i] + B[i];
}

extern "C" __global__
void vec_mul(const float* A, const float* B, float* C, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < N; i += stride) C[i] = A[i] * B[i];
}

extern "C" __global__
void vec_sub(const float* A, const float* B, float* C, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < N; i += stride) C[i] = A[i] - B[i];
}
"""


def run_pipeline_individual(stream, kernels, config, buffers, size, n_iters):
    """Run the 3-stage pipeline `n_iters` times with one launch per stage."""
    add_k, mul_k, sub_k = kernels
    a, b, c, r1, r2, r3 = buffers
    stream.sync()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        launch(
            stream, config, add_k, a.data.ptr, b.data.ptr, r1.data.ptr, np.uint64(size)
        )
        launch(
            stream, config, mul_k, r1.data.ptr, c.data.ptr, r2.data.ptr, np.uint64(size)
        )
        launch(
            stream, config, sub_k, r2.data.ptr, a.data.ptr, r3.data.ptr, np.uint64(size)
        )
    stream.sync()
    return time.perf_counter() - t0


def build_graph(stream, kernels, config, buffers, size):
    """Capture the 3-stage pipeline into a CUDA graph and return it."""
    add_k, mul_k, sub_k = kernels
    a, b, c, r1, r2, r3 = buffers

    graph_builder = stream.create_graph_builder()
    graph_builder.begin_building()
    launch(
        graph_builder,
        config,
        add_k,
        a.data.ptr,
        b.data.ptr,
        r1.data.ptr,
        np.uint64(size),
    )
    launch(
        graph_builder,
        config,
        mul_k,
        r1.data.ptr,
        c.data.ptr,
        r2.data.ptr,
        np.uint64(size),
    )
    launch(
        graph_builder,
        config,
        sub_k,
        r2.data.ptr,
        a.data.ptr,
        r3.data.ptr,
        np.uint64(size),
    )
    graph_builder.end_building()
    graph = graph_builder.complete()
    graph.upload(stream)
    return graph_builder, graph


def run_pipeline_graph(stream, graph, n_iters):
    """Launch the compiled graph `n_iters` times."""
    stream.sync()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        graph.launch(stream)
    stream.sync()
    return time.perf_counter() - t0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="CUDA Graphs demo with cuda.core")
    parser.add_argument(
        "--elements",
        type=int,
        default=1 << 12,
        help="Elements per vector (default: 4096 - small to emphasize launch overhead)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="Number of pipeline iterations to time (default: 1000)",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    args = parser.parse_args()

    device = Device(args.device)
    device.set_current()
    print_gpu_info(device)

    stream = device.create_stream()
    # Tell CuPy to order its allocations on our stream so buffer initialization
    # below is serialized with the kernels we launch.
    cp.cuda.ExternalStream(int(stream.handle)).use()

    graph_builder = graph = None
    try:
        program_options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
        program = Program(PIPELINE_KERNELS, code_type="c++", options=program_options)
        module = program.compile("cubin")
        add_k = module.get_kernel("vec_add")
        mul_k = module.get_kernel("vec_mul")
        sub_k = module.get_kernel("vec_sub")
        kernels = (add_k, mul_k, sub_k)

        N = args.elements
        rng = cp.random.default_rng(seed=0)
        a = rng.random(N, dtype=cp.float32)
        b = rng.random(N, dtype=cp.float32)
        c = rng.random(N, dtype=cp.float32)
        r1 = cp.empty_like(a)
        r2 = cp.empty_like(a)
        r3 = cp.empty_like(a)
        buffers = (a, b, c, r1, r2, r3)

        expected = (a + b) * c - a

        config = LaunchConfig(grid=(N + 255) // 256, block=256)
        device.sync()

        # Warm up compilation/caches, then measure individual launches.
        run_pipeline_individual(stream, kernels, config, buffers, N, n_iters=5)
        t_individual = run_pipeline_individual(
            stream, kernels, config, buffers, N, n_iters=args.iters
        )
        assert cp.allclose(r3, expected, rtol=1e-5, atol=1e-5), (
            "Individual pipeline produced incorrect results"
        )
        print(
            f"\nIndividual launches: {args.iters} iters in {t_individual:.4f}s"
            f"  ({t_individual * 1e6 / args.iters:.2f} us/iter)"
        )

        # Capture the same pipeline as a graph and measure the replay.
        print("\nBuilding CUDA graph...")
        graph_builder, graph = build_graph(stream, kernels, config, buffers, N)

        run_pipeline_graph(stream, graph, n_iters=5)  # warm up
        t_graph = run_pipeline_graph(stream, graph, n_iters=args.iters)
        assert cp.allclose(r3, expected, rtol=1e-5, atol=1e-5), (
            "Graph pipeline produced incorrect results"
        )
        print(
            f"Graph replay:       {args.iters} iters in {t_graph:.4f}s"
            f"  ({t_graph * 1e6 / args.iters:.2f} us/iter)"
        )
        if t_graph > 0:
            print(f"Graph speedup: {t_individual / t_graph:.2f}x")

        # Demonstrate that the graph replays against current buffer contents.
        a[:] = cp.ones(N, dtype=cp.float32)
        b[:] = cp.full(N, 2.0, dtype=cp.float32)
        c[:] = cp.full(N, 3.0, dtype=cp.float32)
        device.sync()
        # r3 = (a + b) * c - a = (1 + 2) * 3 - 1 = 8
        graph.launch(stream)
        stream.sync()
        assert cp.allclose(r3, 8.0), "Graph replay with new data produced wrong result"
        print(
            "\nGraph replay on updated data verified (same graph, new buffer contents)"
        )

        print("\nDone")
        return 0
    finally:
        if graph is not None:
            graph.close()
        if graph_builder is not None:
            graph_builder.close()
        stream.close()
        cp.cuda.Stream.null.use()


if __name__ == "__main__":
    sys.exit(main())
