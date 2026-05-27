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
PyTorch Custom GPU Operator using cuda.core

Question: How do I add a custom GPU op to PyTorch?
Answer: This sample shows the complete workflow.

This sample implements a custom square operation (y = x²) to demonstrate:
- Writing a CUDA kernel
- Compiling with cuda.core
- Integrating with PyTorch's autograd system
- Proper device and stream management
"""

import sys

try:
    import torch
    from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install: pip install torch cuda-python cuda-core")
    sys.exit(1)


# ============================================================================
# Step 1: Define CUDA Kernel
# ============================================================================
# Simple element-wise square: y = x²
# This kernel is easy to understand and verify

SQUARE_KERNEL = """
extern "C" __global__
void square_kernel(const float* x, float* y, int n)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        y[i] = x[i] * x[i];
    }
}
"""


# ============================================================================
# PyTorch Stream Wrapper
# ============================================================================
# cuda.core requires objects with __cuda_stream__ protocol
class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python


# ============================================================================
# Step 2: Kernel Compilation and Caching
# ============================================================================
# Compile kernel once per device and cache it to avoid recompilation overhead
# In real training loops, this avoids paying compilation cost on every forward.


_kernel_cache = {}


def get_square_kernel(device):
    """
    Get or compile the square kernel for a given device.

    Parameters
    ----------
    device : Device
        CUDA device object

    Returns
    -------
    Kernel
        Compiled CUDA kernel
    """
    # Cache key based on device to avoid recompiling for the same GPU
    key = device.pci_bus_id

    if key not in _kernel_cache:
        # Compile the kernel with appropriate architecture
        opts = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
        prog = Program(SQUARE_KERNEL, code_type="c++", options=opts)
        mod = prog.compile("cubin")
        _kernel_cache[key] = mod.get_kernel("square_kernel")

    return _kernel_cache[key]


# ============================================================================
# Step 3: PyTorch Autograd Function
# ============================================================================
# This integrates the CUDA kernel with PyTorch's automatic differentiation


class SquareOp(torch.autograd.Function):
    """
    Custom square operation using cuda.core.

    Forward: y = x² (computed with custom CUDA kernel)
    Backward: grad_x = 2 * x * grad_y (computed with PyTorch)
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass: compute y = x² using custom CUDA kernel.

        Parameters
        ----------
        ctx : Context
            PyTorch context for saving tensors
        x : torch.Tensor
            Input tensor (must be CUDA, float32, contiguous)

        Returns
        -------
        torch.Tensor
            Output tensor with y = x²
        """
        # Validate input requirements
        if not x.is_cuda:
            raise RuntimeError("SquareOp only supports CUDA tensors")
        if x.dtype != torch.float32:
            raise RuntimeError("SquareOp only supports float32 tensors")

        # Ensure contiguous memory layout for efficient kernel access
        x = x.contiguous()

        device = Device()
        # Use PyTorch's current stream to ensure proper ordering with other PyTorch ops
        # Create a cuda.core Stream from PyTorch's stream wrapper
        torch_stream = torch.cuda.current_stream(device=x.device)
        stream = device.create_stream(PyTorchStreamWrapper(torch_stream))

        # Create a try/finally block to ensure the stream is properly closed
        try:
            # Get compiled kernel (cached)
            kernel = get_square_kernel(device)

            # Allocate output tensor
            y = torch.empty_like(x)

            # Configure kernel launch
            n = int(x.numel())
            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            config = LaunchConfig(grid=blocks_per_grid, block=threads_per_block)

            # Launch the kernel
            launch(stream, config, kernel, x.data_ptr(), y.data_ptr(), n)
        finally:
            # Ensure stream is properly closed
            stream.close()

        # Save input for backward pass
        ctx.save_for_backward(x)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradient.

        For y = x², the derivative is dy/dx = 2x
        Therefore: grad_x = grad_output * 2x

        Parameters
        ----------
        ctx : Context
            PyTorch context with saved tensors
        grad_output : torch.Tensor
            Gradient from upstream

        Returns
        -------
        torch.Tensor
            Gradient with respect to input
        """
        # Retrieve saved input
        (x,) = ctx.saved_tensors

        # Note: We assume grad_output has the same dtype and device as x.
        # This is guaranteed by PyTorch's autograd system.

        # Compute gradient: d(x²)/dx = 2x
        grad_x = 2.0 * x * grad_output

        return grad_x


# ============================================================================
# Step 4: Public API
# ============================================================================


def square(x):
    """
    Apply element-wise square operation using custom CUDA kernel.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (must be on CUDA device, dtype=float32)

    Returns
    -------
    torch.Tensor
        Output tensor with y = x²

    Examples
    --------
    >>> x = torch.randn(100, device='cuda')
    >>> y = square(x)
    >>> assert torch.allclose(y, x ** 2)
    """
    return SquareOp.apply(x)


# ============================================================================
# Step 5: Testing and Verification
# ============================================================================


def main():
    """Test the custom square operation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Custom PyTorch Square Operator using cuda.core"
    )
    parser.add_argument(
        "--size", type=int, default=10000, help="Number of elements (default: 10000)"
    )

    args = parser.parse_args()

    # Device info
    device = Device()
    device.set_current()
    major, minor = device.compute_capability

    print("\nDevice Information:")
    print(f"  Name: {device.name}")
    print(f"  Compute Capability: sm_{major}{minor}")

    print("\n" + "=" * 70)
    print("Custom PyTorch Square Operator Test")
    print("=" * 70)

    # ========================================================================
    # Test 1: Forward Pass Correctness
    # ========================================================================
    print("\n" + "-" * 70)
    print("Test 1: Forward Pass")
    print("-" * 70)

    x = torch.randn(args.size, dtype=torch.float32, device="cuda")

    # Custom square operation
    y_custom = square(x)

    # PyTorch reference
    y_reference = x**2

    # Check correctness
    max_error = torch.max(torch.abs(y_custom - y_reference)).item()

    print(f"Max absolute error: {max_error:.2e}")

    if torch.allclose(y_custom, y_reference, rtol=1e-5, atol=1e-6):
        print("[PASS] Forward pass PASSED")
    else:
        print("[FAIL] Forward pass FAILED")
        return 1

    # ========================================================================
    # Test 2: Backward Pass (Gradient) Correctness
    # ========================================================================
    print("\n" + "-" * 70)
    print("Test 2: Backward Pass")
    print("-" * 70)

    # Test with requires_grad
    x_custom = torch.randn(
        args.size, dtype=torch.float32, device="cuda", requires_grad=True
    )
    x_reference = x_custom.clone().detach().requires_grad_(True)

    # Forward pass
    y_custom = square(x_custom)
    y_reference = x_reference**2

    # Create upstream gradient
    grad_output = torch.randn_like(y_custom)

    # Backward pass
    y_custom.backward(grad_output)
    y_reference.backward(grad_output)

    # Check gradients
    max_grad_error = torch.max(torch.abs(x_custom.grad - x_reference.grad)).item()

    print(f"Max gradient error: {max_grad_error:.2e}")

    if torch.allclose(x_custom.grad, x_reference.grad, rtol=1e-5, atol=1e-6):
        print("[PASS] Backward pass PASSED")
    else:
        print("[FAIL] Backward pass FAILED")
        return 1

    # ========================================================================
    # Test 3: Multi-dimensional Tensors
    # ========================================================================
    print("\n" + "-" * 70)
    print("Test 3: Multi-dimensional Tensors")
    print("-" * 70)

    # Test with 2D tensor
    x_2d = torch.randn(100, 100, dtype=torch.float32, device="cuda")
    y_2d_custom = square(x_2d)
    y_2d_reference = x_2d**2

    if torch.allclose(y_2d_custom, y_2d_reference, rtol=1e-5, atol=1e-6):
        print("[PASS] 2D tensor test PASSED")
    else:
        print("[FAIL] 2D tensor test FAILED")
        return 1

    # Test with 3D tensor
    x_3d = torch.randn(10, 20, 30, dtype=torch.float32, device="cuda")
    y_3d_custom = square(x_3d)
    y_3d_reference = x_3d**2

    if torch.allclose(y_3d_custom, y_3d_reference, rtol=1e-5, atol=1e-6):
        print("[PASS] 3D tensor test PASSED")
    else:
        print("[FAIL] 3D tensor test FAILED")
        return 1

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("All tests PASSED!")
    print("=" * 70)
    print("\nYour custom GPU operator is working correctly!")
    print("You can now use it in your PyTorch models like any built-in op.")
    print("\nExample usage:")
    print("  x = torch.randn(100, device='cuda')")
    print("  y = square(x)  # Uses your custom CUDA kernel")
    print("  loss = y.sum()")
    print("  loss.backward()  # Gradients computed automatically")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
