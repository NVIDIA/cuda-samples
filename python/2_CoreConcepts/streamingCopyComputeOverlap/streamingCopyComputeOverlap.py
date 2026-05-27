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
Streaming Copy + Compute Overlap

Demonstrates how to overlap memory transfers with kernel computation using
CUDA streams to maximize GPU utilization.

Uses pure cuda.core APIs:
    - Device, Stream for device and stream management
    - PinnedMemoryResource, DeviceMemoryResource for memory allocation
    - Buffer.copy_to() for async memory copies
    - Program, LaunchConfig, launch for kernel compilation and execution
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))

try:
    import numpy as np
    from cuda.core import (
        Device,
        DeviceMemoryResource,
        EventOptions,
        LaunchConfig,
        PinnedMemoryResource,
        Program,
        ProgramOptions,
        launch,
    )
    from cuda_samples_utils import print_gpu_info
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)


# CUDA Kernel - compute-intensive vector operation (grid-stride loop)
VECTOR_SCALE_KERNEL = r"""
extern "C" __global__
void vector_scale(const float* input, float* output, float scale, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < N; i += stride) {
        float val = input[i] * scale;
        // Add compute work to make kernel non-trivial
        for (int j = 0; j < 50; j++) {
            val = sqrtf(val * val + 1.0f);
        }
        output[i] = val;
    }
}
"""


def buffer_to_numpy(buffer, n_elements):
    """Create numpy view of cuda.core Buffer via DLPack."""
    return np.from_dlpack(buffer).view(np.float32).reshape(n_elements)


def main():
    print("=" * 60)
    print("Streaming Copy + Compute Overlap")
    print("Using pure cuda.core APIs")
    print("=" * 60)

    # Initialize device
    device = Device(0)
    device.set_current()
    print()
    print_gpu_info(device)

    # Compile kernel
    arch = f"sm_{device.arch}"
    program = Program(
        VECTOR_SCALE_KERNEL, code_type="c++", options=ProgramOptions(arch=arch)
    )
    kernel = program.compile(target_type="cubin").get_kernel("vector_scale")
    print("Kernel compiled ✓")

    # Parameters
    N = 16_000_000  # 16M elements
    n_bytes = N * 4
    scale = 2.5
    n_runs = 10

    print(f"\nProblem size: {N:,} elements ({n_bytes / 1024 / 1024:.0f} MB)")

    # Create memory resources
    pinned_mr = PinnedMemoryResource()
    device_mr = DeviceMemoryResource(device.device_id)
    default_stream = device.create_stream()

    # =========================================================================
    # Sequential Execution
    # =========================================================================
    print("\n--- Sequential (no overlap) ---")
    print("Timeline: [H2D][Compute][D2H]")

    h_in = h_out = d_in = d_out = None
    try:
        # Pre-allocate buffers
        h_in = pinned_mr.allocate(n_bytes, default_stream)
        h_out = pinned_mr.allocate(n_bytes, default_stream)
        d_in = device_mr.allocate(n_bytes, default_stream)
        d_out = device_mr.allocate(n_bytes, default_stream)
        # Sync before numpy access (numpy operations aren't stream ordered)
        default_stream.sync()

        # Initialize input
        np_in = buffer_to_numpy(h_in, N)
        np_in[:] = np.random.rand(N).astype(np.float32) * 100

        config = LaunchConfig(grid=((N + 255) // 256,), block=(256,))
        event_opts = EventOptions(enable_timing=True)

        # Warm up
        h_in.copy_to(d_in, stream=default_stream)
        launch(
            default_stream,
            config,
            kernel,
            d_in,
            d_out,
            np.float32(scale),
            np.uint64(N),
        )
        d_out.copy_to(h_out, stream=default_stream)
        default_stream.sync()

        # Benchmark with CUDA events
        times = []
        for _ in range(n_runs):
            start_ev = device.create_event(options=event_opts)
            end_ev = device.create_event(options=event_opts)
            default_stream.record(start_ev)
            h_in.copy_to(d_in, stream=default_stream)  # Async H2D
            launch(
                default_stream,
                config,
                kernel,
                d_in,
                d_out,
                np.float32(scale),
                np.uint64(N),
            )
            d_out.copy_to(h_out, stream=default_stream)  # Async D2H
            default_stream.record(end_ev)
            default_stream.sync()
            times.append(end_ev - start_ev)

        seq_time = np.mean(times)
        print(f"Time: {seq_time:.2f} ms (±{np.std(times):.2f})")

        # Verification: compute expected on CPU and compare
        default_stream.sync()
        np_out = buffer_to_numpy(h_out, N)
        expected = np_in.astype(np.float32) * scale
        for _ in range(50):
            expected = np.sqrt(expected * expected + 1.0).astype(np.float32)
        if np.allclose(np_out, expected, rtol=1e-4, atol=1e-4):
            print("Verification: PASSED")
        else:
            print("Verification: FAILED")
    finally:
        for buf in (h_in, h_out, d_in, d_out):
            if buf is not None:
                buf.close()
        default_stream.close()

    # =========================================================================
    # Streamed Execution
    # =========================================================================
    print("\n--- Streamed (with overlap) ---")
    print("Stream 0: [H2D][Compute][D2H]")
    print("Stream 1:      [H2D][Compute][D2H]")
    print("Stream 2:           [H2D][Compute][D2H]")
    print("...")

    for n_streams in [2, 4, 8]:
        chunk_size = N // n_streams
        chunk_bytes = chunk_size * 4

        # Create streams
        streams = [device.create_stream() for _ in range(n_streams)]

        # Pre-allocate per-stream buffers
        h_ins, h_outs, d_ins, d_outs = [], [], [], []
        try:
            for i in range(n_streams):
                h_ins.append(pinned_mr.allocate(chunk_bytes, streams[i]))
                h_outs.append(pinned_mr.allocate(chunk_bytes, streams[i]))
                d_ins.append(device_mr.allocate(chunk_bytes, streams[i]))
                d_outs.append(device_mr.allocate(chunk_bytes, streams[i]))

            # Initialize input data
            for i in range(n_streams):
                streams[i].sync()
                np_view = buffer_to_numpy(h_ins[i], chunk_size)
                np_view[:] = np.random.rand(chunk_size).astype(np.float32) * 100

            chunk_config = LaunchConfig(grid=((chunk_size + 255) // 256,), block=(256,))

            # Warm up
            for i in range(n_streams):
                h_ins[i].copy_to(d_ins[i], stream=streams[i])
                launch(
                    streams[i],
                    chunk_config,
                    kernel,
                    d_ins[i],
                    d_outs[i],
                    np.float32(scale),
                    np.uint64(chunk_size),
                )
                d_outs[i].copy_to(h_outs[i], stream=streams[i])
            for stream in streams:
                stream.sync()

            # Benchmark with CUDA events (use stream 0 for timing)
            times = []
            event_opts = EventOptions(enable_timing=True)
            for _ in range(n_runs):
                start_ev = device.create_event(options=event_opts)
                end_ev = device.create_event(options=event_opts)
                streams[0].record(start_ev)

                # Issue all operations - they overlap across streams
                for i in range(n_streams):
                    h_ins[i].copy_to(d_ins[i], stream=streams[i])  # Async H2D
                    launch(
                        streams[i],
                        chunk_config,
                        kernel,
                        d_ins[i],
                        d_outs[i],
                        np.float32(scale),
                        np.uint64(chunk_size),
                    )
                    d_outs[i].copy_to(h_outs[i], stream=streams[i])  # Async D2H

                # Wait for all streams, record end on stream 0
                for stream in streams:
                    stream.sync()
                streams[0].record(end_ev)
                streams[0].sync()
                times.append(end_ev - start_ev)

            avg = np.mean(times)
            speedup = seq_time / avg
            print(
                f"{n_streams} streams: {avg:.2f} ms (±{np.std(times):.2f}) "
                f"- speedup: {speedup:.2f}x"
            )

            # Verification (streamed): concatenate chunks and compare to expected
            for s in streams:
                s.sync()
            out_chunks = [
                buffer_to_numpy(h_outs[i], chunk_size) for i in range(n_streams)
            ]
            in_chunks = [
                buffer_to_numpy(h_ins[i], chunk_size) for i in range(n_streams)
            ]
            np_out = np.concatenate(out_chunks)
            np_in = np.concatenate(in_chunks)
            expected = np_in.astype(np.float32) * scale
            for _ in range(50):
                expected = np.sqrt(expected * expected + 1.0).astype(np.float32)
            if not np.allclose(np_out, expected, rtol=1e-4, atol=1e-4):
                print(f"  Verification: FAILED for {n_streams} streams")
        finally:
            for buf in h_ins + h_outs + d_ins + d_outs:
                buf.close()
            for s in streams:
                s.close()

    print("\n" + "=" * 60)
    print("Key: Pinned memory + multiple streams = overlap transfers with compute")
    print("\nNote: Speedup depends on hardware characteristics. This technique")
    print("benefits most when transfer time is significant relative to compute.")
    print("=" * 60)


if __name__ == "__main__":
    main()
