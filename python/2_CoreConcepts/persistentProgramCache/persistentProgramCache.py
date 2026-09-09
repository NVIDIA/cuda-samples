# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
Persistent Program Cache with cuda.core

This sample demonstrates how to persist and reuse compiled CUDA artifacts
with cuda.core. A generated matmul + epilogue kernel is compiled to CUBIN
on a cache miss, stored in a disk-backed FileStreamProgramCache, and loaded
from that cache on later runs.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import cupy as cp
    import numpy as np
    from cuda.core import (
        Device,
        EventOptions,
        LaunchConfig,
        ObjectCode,
        Program,
        ProgramOptions,
        launch,
    )
    from cuda.core.utils import FileStreamProgramCache, make_program_cache_key
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


KERNEL_NAME = "matmul_epilogue"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "cuda-samples" / "persistentProgramCache"
# Tiled matrix multiply followed by bias and an optional ReLU.
KERNEL_TEMPLATE = r"""
#define TILE_SIZE {tile_size}

extern "C" __global__
void {kernel_name}(const float* __restrict__ A,
                   const float* __restrict__ B,
                   const float* __restrict__ bias,
                   float* __restrict__ C,
                   int M, int N, int K)
{{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float acc = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; ++tile) {{
        int a_col = tile * TILE_SIZE + tx;
        int b_row = tile * TILE_SIZE + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {{
            acc += As[ty][i] * Bs[i][tx];
        }}
        __syncthreads();
    }}

    if (row < M && col < N) {{
        float value = acc + bias[col];
        C[row * N + col] = {epilogue_expr};
    }}
}}
"""
EPILOGUE_EXPRESSIONS = {
    "identity": "value",
    "relu": "value > 0.0f ? value : 0.0f",
}


# Return value from compile_or_load_kernel: the launchable kernel plus cache
# status, artifact size, and timings for comparing miss and hit paths.
@dataclass
class CacheResult:
    kernel: object
    status: str
    key_hex: str
    artifact_bytes: int
    cache_lookup_ms: float
    compile_ms: float | None
    cache_store_ms: float | None
    module_load_ms: float

    @property
    def host_preparation_ms(self) -> float:
        """Total cache lookup, compile/store, and module-load time."""
        total = self.cache_lookup_ms + self.module_load_ms
        if self.compile_ms is not None:
            total += self.compile_ms
        if self.cache_store_ms is not None:
            total += self.cache_store_ms
        return total


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def build_kernel_source(tile_size: int, epilogue: str) -> str:
    """Generate a specialized matmul + epilogue CUDA kernel."""
    return KERNEL_TEMPLATE.format(
        tile_size=tile_size,
        kernel_name=KERNEL_NAME,
        epilogue_expr=EPILOGUE_EXPRESSIONS[epilogue],
    )


def cache_clear(cache_dir: Path) -> None:
    with FileStreamProgramCache(cache_dir) as cache:
        cache.clear()


def compile_or_load_kernel(
    source: str,
    device: Device,
    cache_dir: Path,
) -> CacheResult:
    """Compile the generated kernel or load its CUBIN bytes from cache."""
    options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
    program = Program(source, code_type="c++", options=options)
    # For typical use, pass cache=cache to Program.compile(...). This sample
    # uses explicit cache operations so the miss and hit paths are visible.
    key = make_program_cache_key(
        code=source,
        code_type="c++",
        options=options,
        target_type="cubin",
    )
    key_hex = key.hex()

    # This is the core persistent-cache flow: derive the same key for the same
    # source/options/target, load cached CUBIN bytes on a hit, or compile and
    # store the artifact on a miss.
    with FileStreamProgramCache(cache_dir) as cache:
        # These timings are included to compare cache misses and hits.
        start = time.perf_counter()
        cached = cache.get(key)
        cache_lookup_ms = elapsed_ms(start)

        if cached is not None:
            start = time.perf_counter()
            module = ObjectCode.from_cubin(cached, name="persistentProgramCache")
            kernel = module.get_kernel(KERNEL_NAME)
            module_load_ms = elapsed_ms(start)
            return CacheResult(
                kernel=kernel,
                status="HIT",
                key_hex=key_hex,
                artifact_bytes=len(cached),
                cache_lookup_ms=cache_lookup_ms,
                compile_ms=None,
                cache_store_ms=None,
                module_load_ms=module_load_ms,
            )

        start = time.perf_counter()
        module = program.compile("cubin")
        compile_ms = elapsed_ms(start)

        artifact_bytes = len(bytes(module.code))
        start = time.perf_counter()
        cache[key] = module
        cache_store_ms = elapsed_ms(start)

        start = time.perf_counter()
        kernel = module.get_kernel(KERNEL_NAME)
        module_load_ms = elapsed_ms(start)

        return CacheResult(
            kernel=kernel,
            status="MISS",
            key_hex=key_hex,
            artifact_bytes=artifact_bytes,
            cache_lookup_ms=cache_lookup_ms,
            compile_ms=compile_ms,
            cache_store_ms=cache_store_ms,
            module_load_ms=module_load_ms,
        )


def make_host_inputs(m: int, n: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=123)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    bias = rng.standard_normal(n).astype(np.float32)
    return a, b, bias


def host_reference(
    a: np.ndarray,
    b: np.ndarray,
    bias: np.ndarray,
    epilogue: str,
) -> np.ndarray:
    out = a @ b
    out += bias.reshape(1, -1)
    if epilogue == "relu":
        out = np.maximum(out, 0.0)
    return out.astype(np.float32, copy=False)


def print_cache_report(result: CacheResult) -> None:
    print(f"Cache status:       {result.status}")
    print(f"Cache key:          {result.key_hex[:24]}...")
    print(f"Artifact size:      {result.artifact_bytes / 1024:.1f} KiB")
    print(f"Cache lookup:       {result.cache_lookup_ms:.3f} ms")
    if result.compile_ms is None:
        print("Compile time:       skipped")
    else:
        print(f"Compile time:       {result.compile_ms:.3f} ms")
    if result.cache_store_ms is not None:
        print(f"Cache store:        {result.cache_store_ms:.3f} ms")
    print(f"Module load:        {result.module_load_ms:.3f} ms")
    print(f"Host prep time:     {result.host_preparation_ms:.3f} ms")


def run_with_cache(
    device_id: int,
    m: int,
    n: int,
    k: int,
    tile_size: int,
    epilogue: str,
    cache_dir: Path,
    clear_cache: bool,
    warmup: int,
    iterations: int,
) -> bool:
    if clear_cache:
        cache_clear(cache_dir)

    device = Device(device_id)
    device.set_current()
    cp.cuda.Device(device_id).use()

    stream = device.create_stream()
    cp.cuda.Stream.from_external(stream).use()

    try:
        print("\nPersistent Program Cache")
        cc = device.compute_capability
        print(f"Device:             {device.name}")
        print(f"Compute Capability: {cc.major}.{cc.minor}")
        print(f"Cache directory:    {cache_dir}")
        print(f"Workload:           C = {epilogue}(A @ B + bias)")
        print(f"Matrix sizes:       M={m}, N={n}, K={k}")
        print(f"Tile size:          {tile_size}")
        print(f"Timed launches:     warmup={warmup}, iterations={iterations}")
        print()

        source = build_kernel_source(tile_size=tile_size, epilogue=epilogue)
        cache_result = compile_or_load_kernel(
            source=source,
            device=device,
            cache_dir=cache_dir,
        )
        print_cache_report(cache_result)

        host_a, host_b, host_bias = make_host_inputs(m, n, k)
        expected = host_reference(host_a, host_b, host_bias, epilogue)

        d_a = cp.asarray(host_a)
        d_b = cp.asarray(host_b)
        d_bias = cp.asarray(host_bias)
        d_c = cp.empty((m, n), dtype=cp.float32)
        stream.sync()

        grid = ((n + tile_size - 1) // tile_size, (m + tile_size - 1) // tile_size)
        block = (tile_size, tile_size)
        config = LaunchConfig(grid=grid, block=block)
        kernel_args = (
            d_a.data.ptr,
            d_b.data.ptr,
            d_bias.data.ptr,
            d_c.data.ptr,
            np.int32(m),
            np.int32(n),
            np.int32(k),
        )

        for _ in range(warmup):
            launch(stream, config, cache_result.kernel, *kernel_args)
        stream.sync()

        event_options = EventOptions(timing_enabled=True)
        start_event = device.create_event(options=event_options)
        end_event = device.create_event(options=event_options)

        stream.record(start_event)
        for _ in range(iterations):
            launch(stream, config, cache_result.kernel, *kernel_args)
        stream.record(end_event)
        end_event.sync()

        kernel_time_ms = (end_event - start_event) / iterations

        stream.sync()
        actual = cp.asnumpy(d_c)
        max_error = float(np.max(np.abs(actual - expected)))
        ok = bool(np.allclose(actual, expected, rtol=1e-3, atol=1e-3))

        print(f"Kernel time:        {kernel_time_ms:.3f} ms")
        print(f"Max error:          {max_error:.6f}")
        print(f"Validation:         {'PASSED' if ok else 'FAILED'}")

        return ok
    finally:
        cp.cuda.Stream.null.use()
        stream.close()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def cache_dir_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent cuda.core program cache for generated CUDA kernels"
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    parser.add_argument("--m", type=positive_int, default=512, help="Rows of A and C")
    parser.add_argument("--n", type=positive_int, default=512, help="Columns of B and C")
    parser.add_argument("--k", type=positive_int, default=512, help="Columns of A / rows of B")
    parser.add_argument(
        "--tile-size",
        type=int,
        choices=(16, 32),
        default=16,
        help="Compile-time tile size specialization",
    )
    parser.add_argument(
        "--epilogue",
        choices=tuple(EPILOGUE_EXPRESSIONS),
        default="relu",
        help="Compile-time epilogue specialization",
    )
    parser.add_argument(
        "--cache-dir",
        type=cache_dir_arg,
        default=DEFAULT_CACHE_DIR,
        help=f"Program cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument("--clear-cache", action="store_true", help="Clear the cache before running")
    parser.add_argument("--warmup", type=positive_int, default=5, help="Warmup launches")
    parser.add_argument("--iterations", type=positive_int, default=20, help="Timed launches")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ok = run_with_cache(
        device_id=args.device,
        m=args.m,
        n=args.n,
        k=args.k,
        tile_size=args.tile_size,
        epilogue=args.epilogue,
        cache_dir=args.cache_dir,
        clear_cache=args.clear_cache,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
