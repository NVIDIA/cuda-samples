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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS `AS IS'' AND ANY
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
Process Checkpointing Sample using CUDA Core API.

The sample allocates a GPU buffer, fills it with a deterministic
pattern via a kernel, hashes the contents, runs the full
lock/checkpoint/restore/unlock cycle on its own PID, and re-hashes
the buffer afterwards to verify that the GPU memory contents
survived the round trip.
"""

import argparse
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np
from cuda.bindings import driver as cudrv
from cuda.core import (
    Device,
    LaunchConfig,
    Program,
    ProgramOptions,
    checkpoint,
    launch,
)

# Small fill kernel: deterministic, non-trivial pattern so the before/after
# hashes would disagree on any bit flip.
KERNEL_SRC = r"""
extern "C" __global__ void fill_pattern(float *out, unsigned long long n)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float v = (float)(i & 0xFFFFu) * 1e-3f + 1.0f;
        float u = (float)((i >> 16) & 0xFFFFu) * 1e-4f + 0.5f;
        // A handful of dependent ops per element. Deterministic given i.
        for (int k = 0; k < 8; ++k) {
            v = v * 1.000001f + u;
            u = u * 0.999999f + v * 1e-6f;
        }
        out[i] = v + u;
    }
}
"""


@dataclass
class StepTiming:
    label: str
    duration_ms: float
    state_after: str


def _cu_check(result) -> None:
    err = result[0]
    if int(err) != 0:
        raise RuntimeError(f"CUDA driver call failed: {err}")


def compile_fill_kernel(device: Device):
    options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
    program = Program(KERNEL_SRC, code_type="c++", options=options)
    module = program.compile("cubin", name_expressions=("fill_pattern",))
    return module.get_kernel("fill_pattern")


def hash_device_buffer(device_buffer, host: np.ndarray) -> str:
    _cu_check(
        cudrv.cuMemcpyDtoH(
            host.ctypes.data,
            device_buffer.handle,
            host.nbytes,
        )
    )
    return hashlib.sha256(host.tobytes()).hexdigest()[:16]


def _time_call(fn, *args, **kwargs) -> float:
    t0 = time.monotonic()
    fn(*args, **kwargs)
    return (time.monotonic() - t0) * 1000.0


def run_lifecycle(proc: checkpoint.Process, lock_timeout_ms: int) -> List[StepTiming]:
    """
    Drive the full `lock -> checkpoint -> restore -> unlock` cycle on
    `proc` and return per-step timings with the state observed after
    each step.

    Note on state after `restore()`: the driver leaves the process in
    the `locked` state. You must still call `unlock()` to return to
    `running`.
    """
    timings: List[StepTiming] = [StepTiming("initial", 0.0, proc.state)]

    ms = _time_call(proc.lock, timeout_ms=lock_timeout_ms)
    timings.append(StepTiming("lock", ms, proc.state))

    ms = _time_call(proc.checkpoint)
    timings.append(StepTiming("checkpoint", ms, proc.state))

    ms = _time_call(proc.restore)
    timings.append(StepTiming("restore", ms, proc.state))

    ms = _time_call(proc.unlock)
    timings.append(StepTiming("unlock", ms, proc.state))

    return timings


def print_timings(timings: List[StepTiming]) -> None:
    print()
    header = f"{'step':<14}{'duration (ms)':>18}{'state after':>18}"
    print(header)
    print("-" * len(header))
    total = 0.0
    for t in timings:
        if t.label == "initial":
            dur = "-"
        else:
            dur = f"{t.duration_ms:.3f}"
            total += t.duration_ms
        print(f"{t.label:<14}{dur:>18}{t.state_after:>18}")
    print("-" * len(header))
    print(f"{'total':<14}{total:>18.3f}{'':>18}")


def main():
    parser = argparse.ArgumentParser(
        description="CUDA process checkpoint sample using cuda.core",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device ID (default: 0)"
    )
    parser.add_argument(
        "--buffer-mib",
        type=int,
        default=16,
        help="GPU buffer size in MiB (default: 16)",
    )
    parser.add_argument(
        "--lock-timeout-ms",
        type=int,
        default=5000,
        help="Timeout passed to Process.lock in ms (default: 5000)",
    )
    args = parser.parse_args()

    if sys.platform != "linux":
        print("Error: CUDA process checkpointing is Linux-only.")
        return 1

    if args.buffer_mib <= 0:
        print("Error: --buffer-mib must be positive")
        return 1

    print("[Process Checkpoint Sample using CUDA Core API]")
    print(f"PID:                {os.getpid()}")

    device = Device(args.device)
    device.set_current()
    print(f"Device:             {device.name}")
    print(f"Compute Capability: sm_{device.arch}")
    print(f"Buffer size:        {args.buffer_mib} MiB")
    print(f"Lock timeout:       {args.lock_timeout_ms} ms")

    print()
    print("Compiling kernel ...")
    fill_kernel = compile_fill_kernel(device)

    buffer_bytes = args.buffer_mib * 1024 * 1024
    n_elements = buffer_bytes // 4  # float32

    stream = device.create_stream()
    device_buffer = device.memory_resource.allocate(buffer_bytes, stream=stream)
    try:
        print("Writing deterministic pattern to GPU buffer ...")
        block = 256
        grid = (n_elements + block - 1) // block
        cfg = LaunchConfig(grid=grid, block=block)
        launch(stream, cfg, fill_kernel, device_buffer, np.uint64(n_elements))
        stream.sync()

        host = np.empty(n_elements, dtype=np.float32)

        hash_before = hash_device_buffer(device_buffer, host)
        print(f"Buffer hash (before): {hash_before}")

        print()
        print("Running checkpoint lifecycle on self ...")
        proc = checkpoint.Process(os.getpid())
        timings = run_lifecycle(proc, args.lock_timeout_ms)
        print_timings(timings)

        hash_after = hash_device_buffer(device_buffer, host)

        print()
        print(f"Buffer hash (before): {hash_before}")
        print(f"Buffer hash (after):  {hash_after}")

        if hash_before != hash_after:
            print()
            print("FAIL: GPU buffer contents changed across checkpoint/restore.")
            return 1

        print()
        print("PASS: GPU buffer contents survived checkpoint/restore.")
    finally:
        device_buffer.close(stream)

    print()
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
