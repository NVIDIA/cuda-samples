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
cuda.compute: Python lambdas as device-wide operators

This sample demonstrates how cuda.compute 1.0 (from the cuda-cccl
package) accepts plain Python callables, including lambdas, as the
operators that drive device-wide reductions, transforms, and scans.
Internally cuda.compute JIT-compiles the callable with Numba for the
device, so you can iterate on the operator in pure Python and still
get a fused GPU kernel.

The sample exercises three algorithm families with Python lambdas /
regular functions:

  1. cuda.compute.reduce_into - sum via a lambda.
  2. cuda.compute.unary_transform - elementwise y = x*x + 1 via a lambda.
  3. cuda.compute.inclusive_scan - prefix sum over only the even values,
     using a regular Python function as the binary operator.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))

try:
    import cuda.compute
    import cupy as cp
    import numpy as np
    from cuda.core import Device
    from cuda_samples_utils import print_gpu_info  # noqa: E402
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def demo_reduce_lambda() -> bool:
    """reduce_into driven by a lambda."""
    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    d_in = cp.arange(1, 11, dtype=dtype)  # 1..10
    d_out = cp.empty(1, dtype=dtype)

    cuda.compute.reduce_into(
        d_in=d_in,
        d_out=d_out,
        num_items=int(d_in.size),
        op=lambda a, b: a + b,
        h_init=h_init,
    )

    got = int(d_out.get()[0])
    expected = int(d_in.get().sum())
    ok = got == expected
    print(
        f"reduce_into(lambda a,b: a+b) over 1..10 -> {got} "
        f"(expected {expected})  {'OK' if ok else 'FAIL'}"
    )
    return ok


def demo_unary_transform_lambda() -> bool:
    """unary_transform driven by a lambda: y = x*x + 1."""
    d_in = cp.arange(8, dtype=cp.int32)
    d_out = cp.empty_like(d_in)

    cuda.compute.unary_transform(
        d_in=d_in,
        d_out=d_out,
        num_items=int(d_in.size),
        op=lambda x: x * x + 1,
    )

    got = d_out.get()
    expected = (d_in.get().astype(np.int64) ** 2 + 1).astype(np.int32)
    ok = np.array_equal(got, expected)
    print(
        f"unary_transform(lambda x: x*x + 1):\n"
        f"  got      = {got.tolist()}\n"
        f"  expected = {expected.tolist()}  {'OK' if ok else 'FAIL'}"
    )
    return ok


def demo_scan_custom_op() -> bool:
    """inclusive_scan with a Python function that sums only even values.

    This shows the same pattern that also works for reduce/transform:
    the Python callable is JIT-compiled for the device by cuda.compute.
    """
    dtype = np.int32
    d_in = cp.array([1, 2, 3, 4, 5, 6], dtype=dtype)
    d_out = cp.empty_like(d_in)
    h_init = np.array([0], dtype=dtype)

    def add_evens(a, b):
        # Treat odd operands as zero; scan accumulates only even values.
        return (a if a % 2 == 0 else 0) + (b if b % 2 == 0 else 0)

    cuda.compute.inclusive_scan(
        d_in=d_in,
        d_out=d_out,
        op=add_evens,
        init_value=h_init,
        num_items=int(d_in.size),
    )

    got = d_out.get()
    # Host reference: running sum of even-only projection of the input.
    h_in = d_in.get()
    proj = np.where(h_in % 2 == 0, h_in, 0)
    expected = np.cumsum(proj).astype(dtype)
    ok = np.array_equal(got, expected)
    print(
        f"inclusive_scan(add-evens-only) over [1,2,3,4,5,6]:\n"
        f"  got      = {got.tolist()}\n"
        f"  expected = {expected.tolist()}  {'OK' if ok else 'FAIL'}"
    )
    return ok


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Drive cuda.compute device algorithms with Python lambdas / callables"
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    args = parser.parse_args()

    device = Device(args.device)
    device.set_current()
    print_gpu_info(device)
    print()

    ok = True
    ok &= demo_reduce_lambda()
    print()
    ok &= demo_unary_transform_lambda()
    print()
    ok &= demo_scan_custom_op()

    print()
    if ok:
        print("Done")
        return 0
    print("FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
