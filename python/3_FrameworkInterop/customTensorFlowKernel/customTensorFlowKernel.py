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
TensorFlow Custom GPU Operator using cuda.core

Question: How do I add a custom GPU op to TensorFlow?
Answer: This sample shows rapid prototyping with cuda.core + tf.py_function.

This sample implements a custom ReLU operation (y = max(0, x)) to demonstrate:
- Writing CUDA kernels (forward + backward) with grid-stride loops
- Compiling with cuda.core
- Integrating with TensorFlow via tf.py_function
- Proper gradient registration

Dependencies:
- tensorflow: Deep learning framework
- cuda-core: GPU kernel compilation and launch
  (requires >=0.6.0 for LEGACY_DEFAULT_STREAM)
- cuda-python: CUDA driver API bindings
- cupy: Array operations and device pointer access

Note: This approach uses tf.py_function for rapid prototyping. For production
TensorFlow applications, use TensorFlow's C++ Custom Op API.
"""

import sys

try:
    # CuPy is required for array operations and device pointer access
    import cupy as cp
    import tensorflow as tf
    from cuda.core import (
        LEGACY_DEFAULT_STREAM,
        Device,
        LaunchConfig,
        Program,
        ProgramOptions,
        launch,
    )
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install: pip install tensorflow cupy cuda-python cuda-core")
    sys.exit(1)


# ============================================================================
# Step 1: Define CUDA Kernels
# ============================================================================
# Simple element-wise ReLU: y = max(0, x)

RELU_KERNEL = """
extern "C" __global__
void relu_forward_kernel(const float* x, float* y, int n)
{
    // Grid-stride loop: each thread processes multiple elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

extern "C" __global__
void relu_backward_kernel(const float* x, const float* grad_y, float* grad_x, int n)
{
    // Grid-stride loop: each thread processes multiple elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        grad_x[i] = x[i] > 0.0f ? grad_y[i] : 0.0f;
    }
}
"""


# ============================================================================
# Step 2: Kernel Compilation and Caching
# ============================================================================
# Compile kernel once per device and cache it to avoid recompilation overhead
# In real training loops, this avoids paying compilation cost on every forward.

_kernel_cache = {}


def _get_relu_kernels(device):
    """
    Get or compile the ReLU kernels for a given device.

    Parameters
    ----------
    device : Device
        CUDA device object

    Returns
    -------
    tuple
        (forward_kernel, backward_kernel) compiled CUDA kernels
    """
    # Cache key based on device to avoid recompiling for the same GPU
    key = device.pci_bus_id

    if key not in _kernel_cache:
        # Compile the kernel with appropriate architecture
        opts = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
        prog = Program(RELU_KERNEL, code_type="c++", options=opts)
        mod = prog.compile("cubin")
        forward_kernel = mod.get_kernel("relu_forward_kernel")
        backward_kernel = mod.get_kernel("relu_backward_kernel")
        _kernel_cache[key] = (forward_kernel, backward_kernel)

    return _kernel_cache[key]


def _launch_relu_forward(x_np):
    """
    Internal function: Launch forward CUDA kernel.

    Takes numpy array, returns numpy array.
    Uses CuPy for array operations and device pointer access, cuda.core for
    device/stream management.

    Note: LEGACY_DEFAULT_STREAM doesn't require explicit cleanup, but kernel
    launch failures should be handled by the caller. CuPy arrays are
    automatically cleaned up when they go out of scope.
    """
    device = Device()

    # Ensure this device is current (TensorFlow usually does this already)
    device.set_current()

    # Get compiled kernel (cached)
    forward_kernel, _ = _get_relu_kernels(device)

    # Convert numpy to CuPy (CPU-to-GPU copy)
    # CuPy is used for array operations and getting device pointers
    x_cp = cp.asarray(x_np)
    y_cp = cp.empty_like(x_cp)

    # Configure kernel launch
    n = int(x_cp.size)
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    config = LaunchConfig(grid=blocks_per_grid, block=threads_per_block)

    # Launch on the legacy default stream (stream 0) for TensorFlow interop
    launch(
        LEGACY_DEFAULT_STREAM, config, forward_kernel, x_cp.data.ptr, y_cp.data.ptr, n
    )

    # Return as numpy array (GPU-to-CPU copy via cp.asnumpy)
    return cp.asnumpy(y_cp)


def _launch_relu_backward(x_np, grad_y_np):
    """
    Internal function: Launch backward CUDA kernel.

    Takes numpy arrays, returns numpy array.
    Uses CuPy for array operations and device pointer access, cuda.core for
    device/stream management.

    Note: LEGACY_DEFAULT_STREAM doesn't require explicit cleanup, but kernel
    launch failures should be handled by the caller. CuPy arrays are
    automatically cleaned up when they go out of scope.
    """
    device = Device()

    # Ensure this device is current (TensorFlow usually does this already)
    device.set_current()

    # Get compiled kernel (cached)
    _, backward_kernel = _get_relu_kernels(device)

    # Convert numpy to CuPy (CPU-to-GPU copy)
    # CuPy is used for array operations and getting device pointers
    x_cp = cp.asarray(x_np)
    grad_y_cp = cp.asarray(grad_y_np)
    grad_x_cp = cp.empty_like(x_cp)

    # Configure kernel launch
    n = int(x_cp.size)
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    config = LaunchConfig(grid=blocks_per_grid, block=threads_per_block)

    # Launch on the legacy default stream (stream 0) for TensorFlow interop
    launch(
        LEGACY_DEFAULT_STREAM,
        config,
        backward_kernel,
        x_cp.data.ptr,
        grad_y_cp.data.ptr,
        grad_x_cp.data.ptr,
        n,
    )

    # Return as numpy array (GPU-to-CPU copy via cp.asnumpy)
    return cp.asnumpy(grad_x_cp)


# ============================================================================
# Step 3: TensorFlow Integration via tf.py_function
# ============================================================================


@tf.custom_gradient
def custom_relu(x):
    """
    Custom ReLU operation using cuda.core.

    This function provides a TensorFlow-native interface to custom CUDA kernels
    compiled with cuda.core. The implementation uses tf.py_function internally
    to bridge TensorFlow and cuda.core.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor (must be float32 on GPU)

    Returns
    -------
    tf.Tensor
        Output tensor with ReLU applied

    Examples
    --------
    >>> x = tf.random.normal([100], dtype=tf.float32)
    >>> y = custom_relu(x)
    >>> # Use in models
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Dense(128),
    ...     tf.keras.layers.Lambda(custom_relu),  # Custom ReLU
    ...     tf.keras.layers.Dense(10)
    ... ])
    """
    # Validate input
    if x.dtype != tf.float32:
        raise ValueError("custom_relu only supports float32 tensors")

    # Forward pass using tf.py_function
    # py_function allows us to call arbitrary Python code (including cuda.core)
    y = tf.py_function(func=_launch_relu_forward, inp=[x], Tout=tf.float32)

    # Restore shape information (py_function loses shape)
    y.set_shape(x.shape)

    # Define gradient function
    def grad_fn(grad_y):
        """Backward pass using custom CUDA kernel"""
        grad_x = tf.py_function(
            func=_launch_relu_backward, inp=[x, grad_y], Tout=tf.float32
        )
        grad_x.set_shape(x.shape)
        return grad_x

    return y, grad_fn


# ============================================================================
# Step 4: Testing and Verification
# ============================================================================


def main():
    """Test the custom ReLU operation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Custom TensorFlow ReLU Operator using cuda.core"
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
    print(f"  Compute Capability: sm_{major}.{minor}")

    print("\n" + "=" * 70)
    print("Custom TensorFlow ReLU Operator Test")
    print("=" * 70)

    # ========================================================================
    # Test 1: Forward Pass Correctness
    # ========================================================================
    print("\n" + "-" * 70)
    print("Test 1: Forward Pass")
    print("-" * 70)

    # Run on the first visible GPU (respects CUDA_VISIBLE_DEVICES),
    # aligning with cuda.core Device().
    with tf.device("/GPU:0"):
        x = tf.random.normal([args.size], dtype=tf.float32)

        # Custom ReLU operation
        y_custom = custom_relu(x)

        # TensorFlow reference
        y_reference = tf.nn.relu(x)

        # Check correctness
        max_error = tf.reduce_max(tf.abs(y_custom - y_reference)).numpy()

        print(f"Max absolute error: {max_error:.2e}")

        if tf.reduce_all(tf.abs(y_custom - y_reference) < 1e-5):
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

    with tf.device("/GPU:0"):
        x_custom = tf.random.normal([args.size], dtype=tf.float32)
        x_reference = tf.identity(x_custom)

        # Compute gradients with GradientTape
        with tf.GradientTape() as tape_custom:
            tape_custom.watch(x_custom)
            y_custom = custom_relu(x_custom)
        grad_custom = tape_custom.gradient(y_custom, x_custom)

        with tf.GradientTape() as tape_reference:
            tape_reference.watch(x_reference)
            y_reference = tf.nn.relu(x_reference)
        grad_reference = tape_reference.gradient(y_reference, x_reference)

        # Check gradients
        max_grad_error = tf.reduce_max(tf.abs(grad_custom - grad_reference)).numpy()

        print(f"Max gradient error: {max_grad_error:.2e}")

        if tf.reduce_all(tf.abs(grad_custom - grad_reference) < 1e-5):
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

    with tf.device("/GPU:0"):
        # Test with 2D tensor
        x_2d = tf.random.normal([100, 100], dtype=tf.float32)
        y_2d_custom = custom_relu(x_2d)
        y_2d_reference = tf.nn.relu(x_2d)

        if tf.reduce_all(tf.abs(y_2d_custom - y_2d_reference) < 1e-5):
            print("[PASS] 2D tensor test PASSED")
        else:
            print("[FAIL] 2D tensor test FAILED")
            return 1

        # Test with 3D tensor
        x_3d = tf.random.normal([10, 20, 30], dtype=tf.float32)
        y_3d_custom = custom_relu(x_3d)
        y_3d_reference = tf.nn.relu(x_3d)

        if tf.reduce_all(tf.abs(y_3d_custom - y_3d_reference) < 1e-5):
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
    print("You can now use it in your TensorFlow models.")
    print("\nExample usage:")
    print("  x = tf.random.normal([100], dtype=tf.float32)")
    print("  y = custom_relu(x)  # Uses your custom CUDA kernel")
    print("  ")
    print("  # In a model:")
    print("  model = tf.keras.Sequential([")
    print("      tf.keras.layers.Dense(128),")
    print("      tf.keras.layers.Lambda(custom_relu),")
    print("      tf.keras.layers.Dense(10)")
    print("  ])")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
