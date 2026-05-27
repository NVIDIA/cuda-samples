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
This sample demonstrates the parallel binary-search algorithms exposed
by cuda.compute (from the cuda-cccl package). Given a sorted
``d_data`` array and a batch of ``d_values`` to locate, cuda.compute:

  - ``cuda.compute.lower_bound(d_data, num_items, d_values, num_values, d_out)``
    writes, for each value, the lowest index where it could be inserted
    into d_data without breaking the sort order. Matches
    ``numpy.searchsorted(..., side="left")``.

  - ``cuda.compute.upper_bound(d_data, num_items, d_values, num_values, d_out)``
    is the analogous upper form, matching ``side="right"``.

The sample runs both algorithms on a curated sorted input with
duplicates so the lower/upper distinction is visible, verifies the
results against ``numpy.searchsorted``, and prints both sets of
indices side-by-side.
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


def run_binary_search(h_data: np.ndarray, h_values: np.ndarray) -> bool:
    d_data = cp.asarray(h_data)
    d_values = cp.asarray(h_values)

    d_lb = cp.empty(len(h_values), dtype=np.uintp)
    d_ub = cp.empty(len(h_values), dtype=np.uintp)

    cuda.compute.lower_bound(
        d_data=d_data,
        num_items=len(d_data),
        d_values=d_values,
        num_values=len(d_values),
        d_out=d_lb,
    )
    cuda.compute.upper_bound(
        d_data=d_data,
        num_items=len(d_data),
        d_values=d_values,
        num_values=len(d_values),
        d_out=d_ub,
    )

    got_lb = cp.asnumpy(d_lb)
    got_ub = cp.asnumpy(d_ub)
    expected_lb = np.searchsorted(h_data, h_values, side="left").astype(np.uintp)
    expected_ub = np.searchsorted(h_data, h_values, side="right").astype(np.uintp)

    ok_lb = np.array_equal(got_lb, expected_lb)
    ok_ub = np.array_equal(got_ub, expected_ub)

    print(f"  data    = {h_data.tolist()}")
    print(f"  values  = {h_values.tolist()}")
    print(
        f"  lower_bound: got {got_lb.tolist()}  "
        f"expected {expected_lb.tolist()}  {'OK' if ok_lb else 'FAIL'}"
    )
    print(
        f"  upper_bound: got {got_ub.tolist()}  "
        f"expected {expected_ub.tolist()}  {'OK' if ok_ub else 'FAIL'}"
    )
    return ok_lb and ok_ub


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parallel upper_bound / lower_bound via cuda.compute"
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    args = parser.parse_args()

    device = Device(args.device)
    device.set_current()
    print_gpu_info(device)
    print()

    ok = True

    # Case 1: values both inside and outside the data range; no duplicates
    # in the data. lower_bound and upper_bound agree on values not present.
    print("Case 1: distinct data, mixed queries")
    h_data1 = np.array([1, 3, 5, 7, 9], dtype=np.int32)
    h_values1 = np.array([0, 3, 4, 10], dtype=np.int32)
    ok &= run_binary_search(h_data1, h_values1)
    print()

    # Case 2: duplicates in the data so lower_bound and upper_bound diverge
    # on present values.
    print("Case 2: duplicates in data")
    h_data2 = np.array([1, 3, 3, 5, 7, 9], dtype=np.int32)
    h_values2 = np.array([3, 3, 5, 8], dtype=np.int32)
    ok &= run_binary_search(h_data2, h_values2)

    print()
    if ok:
        print("Done")
        return 0
    print("FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
