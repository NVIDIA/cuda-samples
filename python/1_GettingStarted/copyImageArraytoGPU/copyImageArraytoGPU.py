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
Image Array Copy to GPU using CUDA Core API

This sample demonstrates how to copy image arrays between CPU and GPU memory
using NVIDIA's CUDA Core Python API with optimal performance.
"""

import sys
from pathlib import Path

# Add parent directory to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))
from cuda_samples_utils import verify_array_result

try:
    import cupy as cp
    import numpy as np
    from cuda.core import Buffer, Device, PinnedMemoryResource, Stream
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


# ----------------------------- Helper Functions ------------------------------


def make_random_image(h: int, w: int, c: int, dtype=np.uint8) -> np.ndarray:
    """
    Create a random test image for demonstration.

    Args:
        h: Image height in pixels
        w: Image width in pixels
        c: Number of channels (e.g., 3 for RGB)
        dtype: NumPy data type (e.g., np.uint8 for 0-255 pixel values)

    Returns:
        A contiguous NumPy array representing the image
    """
    img = np.random.randint(0, 256, size=(h, w, c), dtype=dtype)
    return np.ascontiguousarray(img)  # Ensure memory is contiguous for GPU transfer


# ----------------------------- Core GPU Functions ---------------------------


def copy_image_to_gpu_cuda_core(
    host_np: np.ndarray, dev: Device, stream: Stream
) -> tuple[Buffer, Buffer]:
    """
    Copy image from CPU memory to GPU memory using optimal transfer method.

    This function demonstrates the recommended approach:
    1. Use pinned memory for faster transfers
    2. Use DLPack for zero-copy array views
    3. Perform async transfers on a CUDA stream

    Args:
        host_np: NumPy array containing image data on CPU
        dev: CUDA device object
        stream: CUDA stream for async operations

    Returns:
        Tuple of (device_buffer, pinned_buffer) - both need to be cleaned up later
    """
    nbytes = host_np.nbytes  # Calculate total bytes needed

    # Step 1: Set up memory resources
    # Device memory resource - allocates on GPU
    device_mr = dev.memory_resource
    # Pinned memory resource - allocates CPU memory that GPU can access faster
    pinned_mr = PinnedMemoryResource()

    # Step 2: Allocate memory buffers
    pinned_buffer = pinned_mr.allocate(nbytes, stream=stream)  # Fast CPU memory
    device_buffer = device_mr.allocate(nbytes, stream=stream)  # GPU memory

    # Step 3: Create a NumPy view of pinned memory using DLPack
    # This allows us to work with pinned memory as if it's a regular NumPy array
    pinned_view = (
        np.from_dlpack(pinned_buffer).view(dtype=host_np.dtype).reshape(host_np.shape)
    )

    # Step 4: Copy image data from regular CPU memory to pinned CPU memory
    # This is a CPU-to-CPU copy, so it's very fast
    np.copyto(pinned_view, host_np)

    # Step 5: Copy from pinned CPU memory to GPU memory
    # This is the actual CPU-to-GPU transfer, done asynchronously
    pinned_buffer.copy_to(device_buffer, stream=stream)

    return device_buffer, pinned_buffer


def copy_image_from_gpu_cuda_core(
    device_buffer: Buffer, shape: tuple, dtype: type, dev: Device, stream: Stream
) -> np.ndarray:
    """
    Copy image from GPU memory back to CPU memory.

    This function reverses the GPU-to-CPU transfer process:
    1. Allocate pinned CPU memory for fast transfer
    2. Copy from GPU to pinned CPU memory
    3. Create NumPy view and copy to regular CPU memory

    Args:
        device_buffer: GPU buffer containing image data
        shape: Original image shape tuple (height, width, channels)
        dtype: Original image data type
        dev: CUDA device object
        stream: CUDA stream for async operations

    Returns:
        NumPy array with image data copied from GPU
    """
    nbytes = np.prod(shape) * np.dtype(dtype).itemsize  # Calculate total bytes

    # Step 1: Create pinned memory for fast GPU-to-CPU transfer
    pinned_mr = PinnedMemoryResource()
    pinned_buffer = pinned_mr.allocate(nbytes, stream=stream)

    # Step 2: Copy from GPU memory to pinned CPU memory
    device_buffer.copy_to(pinned_buffer, stream=stream)
    stream.sync()  # Wait for the GPU transfer to complete

    # Step 3: Create NumPy view of pinned memory using DLPack
    pinned_view = np.from_dlpack(pinned_buffer).view(dtype=dtype).reshape(shape)

    # Step 4: Copy from pinned CPU memory to regular CPU memory
    # This creates the final result that can be used normally
    host_result = pinned_view.copy()

    # Step 5: Clean up the temporary pinned buffer
    pinned_buffer.close(stream)

    return host_result


# ------------------------------ Main Demo ------------------------------------


def main():
    """
    Complete demonstration of GPU image copying workflow.

    This example shows:
    1. Setting up CUDA device and stream
    2. Creating a sample image
    3. Copying image to GPU
    4. Accessing GPU data with CuPy (optional)
    5. Copying image back from GPU
    6. Verifying data integrity
    7. Proper cleanup of resources
    """
    print("[Image Array Copy to GPU using CUDA Core API]")

    # Image parameters - modify these to test different sizes
    H, W, C = 256, 256, 3  # Height=256, Width=256, Channels=3 (RGB)
    dtype = np.uint8  # Standard image pixel type (0-255 values)

    # Step 1: Set up CUDA device and stream
    dev = Device()  # Get default CUDA device (GPU 0)
    dev.set_current()  # Make this device the active one
    stream = dev.create_stream()  # Create stream for async operations

    print(f"Device: {dev.name}")
    print(f"[Image array copy of {H}x{W}x{C} image]")

    # Step 2: Configure CuPy to use our CUDA stream (for interoperability)
    cp.cuda.ExternalStream(int(stream.handle)).use()

    # Step 3: Create a test image on CPU
    print("Creating sample image...")
    host_np = make_random_image(H, W, C, dtype=dtype)

    # Step 4: Copy image from CPU to GPU
    print("Copying image to GPU...")
    device_buffer, pinned_buffer = copy_image_to_gpu_cuda_core(host_np, dev, stream)

    # Step 5: (Optional) Get a CuPy view of GPU data for processing
    # This shows how you can work with the GPU data without copying it back
    print("Creating CuPy view of GPU data...")
    device_cp = cp.from_dlpack(device_buffer).view(dtype=dtype).reshape(H, W, C)

    # Example: compute mean pixel value on GPU
    mean_value = float(cp.mean(device_cp))
    print(f"Mean pixel value (computed on GPU): {mean_value:.2f}")

    # Step 6: Copy image back from GPU to CPU
    print("Copying image back from GPU...")
    host_back = copy_image_from_gpu_cuda_core(
        device_buffer, host_np.shape, host_np.dtype, dev, stream
    )

    # Step 7: Verify that the data survived the round trip
    print("Verifying result...")
    host_back_cp = cp.asarray(host_back)
    host_np_cp = cp.asarray(host_np)
    verify_array_result(host_back_cp, host_np_cp, rtol=0, atol=0)

    # Step 8: Clean up all allocated resources
    device_buffer.close(stream)  # Free GPU memory
    pinned_buffer.close(stream)  # Free pinned CPU memory
    stream.close()  # Close CUDA stream
    cp.cuda.Stream.null.use()  # Reset CuPy's stream to default

    print("\nDone")


if __name__ == "__main__":
    main()
