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

import argparse
import contextlib
import sys
import time
from pathlib import Path

try:
    import cupy as cp
    import numpy as np
    from cuda.core import Device, EventOptions
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Add parent directory to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))
from cuda_samples_utils import verify_array_result


@contextlib.contextmanager
def timer(message):
    """CPU timing context manager."""
    start = time.time()
    yield
    end = time.time()
    print(f"{message}:  {(end - start):.6f} seconds")


@contextlib.contextmanager
def gpu_timer(message, stream):
    """GPU timing context manager using cuda.core CUDA events."""
    event_options = EventOptions(enable_timing=True)
    start_event = stream.record(options=event_options)
    yield
    end_event = stream.record(options=event_options)
    end_event.sync()

    elapsed_time_ms = end_event - start_event  # Returns milliseconds
    elapsed_time_s = elapsed_time_ms / 1000.0  # Convert to seconds
    print(f"{message}:  {elapsed_time_s:.6f} seconds")


def warmup():
    # Pre-runs a simple GPU operation to avoid first-run overhead in benchmarking.
    print("Warmup...")
    a_cp = cp.ones((16, 16))
    b_cp = cp.ones((16, 16))
    result_cp = cp.dot(a_cp, b_cp)
    return result_cp


def run(n):
    # Benchmarks NumPy vs. CuPy matrix multiplication for n x n random arrays.
    # Prints timing results.

    device = Device()  # Use device 0 explicitly
    device.set_current()
    major, minor = device.compute_capability
    print()
    print(f"Device Name: {device.name}, SM: {major}.{minor}")
    print()

    # Create explicit stream
    stream = device.create_stream()

    try:
        # Warm up GPU before measuring
        warmup()
        stream.sync()

        # Generate random matrices on CPU
        a_np = np.random.rand(n, n)
        b_np = np.random.rand(n, n)

        # NumPy dot product (CPU)
        with timer(f"NumPy dot of {n}*{n} arrays"):
            result_np = np.dot(a_np, b_np)

        # Transfer NumPy arrays to GPU (using events for timing)
        with gpu_timer("Transfer arrays to GPU", stream):
            a_cp = cp.asarray(a_np)
            b_cp = cp.asarray(b_np)

        # CuPy dot product (GPU) - using events for accurate GPU timing
        with gpu_timer(f"CuPy dot of {n}*{n} arrays", stream):
            result_cp = cp.dot(a_cp, b_cp)

        print()
        # Result validation
        if not verify_array_result(result_np, result_cp.get()):
            print(
                "Validation FAILED: NumPy and CuPy results do not match "
                "within tolerance"
            )
            sys.exit(1)

        print("Validation PASSED: NumPy and CuPy results match within tolerance")
    finally:
        stream.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_size", "-n", default=4096, type=int, help="Size of the matrix(n * n)."
    )
    args = parser.parse_args()
    run(args.n_size)
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
