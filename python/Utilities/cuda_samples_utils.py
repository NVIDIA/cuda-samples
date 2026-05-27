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

"""
Common CUDA utilities for Python samples.

This module provides common utility functions for CUDA samples including:
- Package requirements checking
- Result verification
- GPU device information

Requirements:
- Python 3.10+
- CUDA Toolkit 13.0+ (recommended; matches cuda-python 13.x)
- cuda-python >= 13.0.0
- cuda-core >= 0.6.0
- cupy-cuda13x >= 13.0.0
- numpy >= 2.3.2 (when used with samples that install it)
"""


def check_cuda_requirements() -> bool:
    """
    Check if required CUDA packages are available.

    Returns
    -------
    bool
        True if requirements are met, False otherwise
    """
    try:
        import cupy as cp  # noqa: F401
        from cuda.core import Device  # noqa: F401

        return True
    except ImportError as e:
        print(f"Error: Required package not found: {e}")
        print("Please install from requirements.txt:")
        print("  pip install -r requirements.txt")
        return False


def verify_array_result(
    result, expected, rtol: float = 1e-5, atol: float = 1e-8, verbose: bool = True
) -> bool:
    """
    Verify that computed result matches expected result.

    Automatically detects whether arrays are NumPy or CuPy and uses the
    appropriate library without unnecessary data transfers.

    Parameters
    ----------
    result : numpy.ndarray or cupy.ndarray
        Computed result array.
    expected : numpy.ndarray or cupy.ndarray
        Expected result array.
    rtol : float
        Relative tolerance (default: 1e-5)
    atol : float
        Absolute tolerance (default: 1e-8)
    verbose : bool
        Whether to print verification result (default: True).

    Returns
    -------
    bool
        True if results match, False otherwise.

    Raises
    ------
    TypeError
        If arrays are not both NumPy or both CuPy, or if CuPy is needed
        but not available.
    """
    import numpy as np

    is_np = isinstance(result, np.ndarray) and isinstance(expected, np.ndarray)

    if is_np:
        allclose = np.allclose
        abs_ = np.abs
        max_ = np.max
    else:
        import cupy as cp

        is_cp = isinstance(result, cp.ndarray) and isinstance(expected, cp.ndarray)

        if not is_cp:
            raise TypeError(
                "verify_array_result expects both arrays to be either "
                "numpy.ndarray or cupy.ndarray"
            )

        allclose = cp.allclose
        abs_ = cp.abs
        max_ = cp.max

    if allclose(result, expected, rtol=rtol, atol=atol):
        if verbose:
            print("Test PASSED")
        return True
    else:
        max_error = max_(abs_(result - expected))
        if verbose:
            print(f"Test FAILED - Max error: {max_error}")
        return False


def print_gpu_info(device) -> None:
    """
    Print GPU device information.

    Parameters
    ----------
    device : cuda.core.Device
        CUDA device object
    """
    print(f"Device: {device.name}")
    cc = device.compute_capability
    print(f"Compute Capability: {cc.major}.{cc.minor}")
