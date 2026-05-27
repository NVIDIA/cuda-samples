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
Prefix Sum (Scan)

Demonstrates parallel prefix sum algorithms using cuda.compute:
- Inclusive scan: output[i] = [init_value] + input[0] + ... + input[i]
- Exclusive scan: output[i] = init_value + input[0] + ... + input[i-1]

Uses cuda.compute APIs for optimized CUB-based scan operations.
Uses cuda.core APIs for device and stream management.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))

try:
    import cupy as cp
    import numpy as np
    from cuda.compute import OpKind, exclusive_scan, inclusive_scan
    from cuda.core import Device, EventOptions
    from cuda_samples_utils import print_gpu_info, verify_array_result
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def main() -> bool:
    """Run prefix sum sample. Returns True if all tests passed."""
    print("=" * 60)
    print("Prefix Sum (Scan) - Using cuda.compute")
    print("=" * 60)

    device = Device(0)
    device.set_current()
    stream = device.create_stream()
    cp_stream = cp.cuda.ExternalStream(int(stream.handle))

    ok = True
    try:
        print()
        print_gpu_info(device)

        h_input = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32)
        init_value = np.array([0], dtype=np.int32)

        # =========================================================================
        # Inclusive Scan
        # =========================================================================
        print("\n" + "-" * 60)
        print("INCLUSIVE SCAN")
        print("-" * 60)
        print(
            "Formula: output[i] = [init_value] + input[0] + input[1] + ... + input[i]"
        )

        with cp_stream:
            d_input = cp.asarray(h_input)
            d_output = cp.empty_like(d_input)

        print(f"\nInput:  {h_input.tolist()}")

        inclusive_scan(
            d_in=d_input,
            d_out=d_output,
            op=OpKind.PLUS,
            init_value=None,
            num_items=len(h_input),
            stream=stream,
        )
        stream.sync()
        print(f"Output: {cp.asnumpy(d_output).tolist()}")

        with cp_stream:
            expected = cp.asarray(np.cumsum(h_input))
        ok &= verify_array_result(d_output, expected, rtol=0, atol=0)

        # =========================================================================
        # Exclusive Scan
        # =========================================================================
        print("\n" + "-" * 60)
        print("EXCLUSIVE SCAN")
        print("-" * 60)
        print("Formula: output[i] = init_value + input[0] + ... + input[i-1]")

        with cp_stream:
            d_output = cp.empty_like(d_input)

        print(f"\nInput:  {h_input.tolist()}")

        exclusive_scan(
            d_in=d_input,
            d_out=d_output,
            op=OpKind.PLUS,
            init_value=init_value,
            num_items=len(h_input),
            stream=stream,
        )
        stream.sync()
        print(f"Output: {cp.asnumpy(d_output).tolist()}")

        with cp_stream:
            expected = cp.asarray(np.concatenate([init_value, np.cumsum(h_input)[:-1]]))
        ok &= verify_array_result(d_output, expected, rtol=0, atol=0)

        # =========================================================================
        # Large Array Performance
        # =========================================================================
        print("\n" + "-" * 60)
        print("PERFORMANCE (10M elements)")
        print("-" * 60)

        N = 10_000_000
        with cp_stream:
            d_large_in = cp.ones(N, dtype=np.int32)
            d_large_out = cp.empty_like(d_large_in)

        inclusive_scan(
            d_in=d_large_in,
            d_out=d_large_out,
            op=OpKind.PLUS,
            init_value=None,
            num_items=N,
            stream=stream,
        )
        stream.sync()

        event_opts = EventOptions(enable_timing=True)
        start_event = device.create_event(options=event_opts)
        end_event = device.create_event(options=event_opts)

        num_iterations = 10
        stream.record(start_event)
        for _ in range(num_iterations):
            inclusive_scan(
                d_in=d_large_in,
                d_out=d_large_out,
                op=OpKind.PLUS,
                init_value=None,
                num_items=N,
                stream=stream,
            )
        stream.record(end_event)
        end_event.sync()
        elapsed_ms = (end_event - start_event) / num_iterations

        print(f"Inclusive scan: {elapsed_ms:.3f} ms")
        print(f"Throughput: {N / elapsed_ms / 1e6:.1f} M elements/ms")

        # =========================================================================
        # Summary
        # =========================================================================
        print("\n" + "=" * 60)
        print("KEY CONCEPTS")
        print("=" * 60)
        print("• Inclusive: output[i] includes input[i]")
        print("• Exclusive: output[i] excludes input[i], starts with init_value")
        print("• cuda.compute provides CUB-based optimized implementations")
        print("• cuda.core Stream integrates with CuPy via ExternalStream")
        print("• Applications: stream compaction, radix sort, histograms")
        print("=" * 60)
        return ok
    finally:
        cp.cuda.Stream.null.use()
        stream.close()


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
