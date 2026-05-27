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
Image Blur with Unified Memory using cuda.core

Demonstrates GPU image blurring using cuda.core APIs for kernel compilation,
launch, and unified memory allocation.
"""

import sys

try:
    import numpy as np
    from cuda.core import (
        Device,
        LaunchConfig,
        ManagedMemoryResource,
        ManagedMemoryResourceOptions,
        Program,
        ProgramOptions,
        launch,
    )
    from PIL import Image
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


# CUDA kernel source code - compiled at runtime by cuda.core.Program
BOX_BLUR_KERNEL_CODE = r"""
extern "C" __global__
void box_blur_3x3(const float* __restrict__ src,
                  float* __restrict__ dst, int H, int W) {
    /*
     * Simple 3x3 box blur CUDA kernel.
     *
     * Each thread computes one output pixel by averaging
     * the 3x3 neighborhood of input pixels (stencil pattern).
     */

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    float sum = 0.0f;
    int count = 0;

    // 3x3 stencil: iterate over neighborhood
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            // Boundary check (clamp to edge)
            if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                sum += src[ny * W + nx];
                count++;
            }
        }
    }

    dst[y * W + x] = sum / count;
}
"""


def make_test_image(h: int, w: int, dtype=np.uint8) -> np.ndarray:
    """Create a test grayscale image for demonstration."""
    img = np.zeros((h, w), dtype=dtype)

    # Create horizontal stripes
    for i in range(0, h, 50):
        img[i : i + 25, :] = 255

    # Create vertical stripes with different intensity
    for j in range(0, w, 50):
        img[:, j : j + 25] = 128

    # Add circular pattern for interesting blur effects
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= (min(h, w) // 6) ** 2
    img[circle_mask] = 200

    return np.ascontiguousarray(img)


def blur_image_unified_memory(
    host_np: np.ndarray, device: Device, stream, kernel
) -> tuple[np.ndarray, object, object]:
    """
    Blur image on GPU using unified memory with cuda.core.

    This function demonstrates:
    1. Allocate managed memory using ManagedMemoryResource
    2. Create zero-copy numpy views using np.from_dlpack()
    3. Launch kernel via cuda.core.launch

    Args:
        host_np: NumPy array containing image data on CPU
        device: CUDA device to use
        stream: cuda.core Stream for async operations
        kernel: Compiled cuda.core Kernel object

    Returns:
        Tuple of (dst_np, src_buf, dst_buf). dst_np is a zero-copy view into
        unified memory. Caller must close src_buf and dst_buf when done with
        dst_np to avoid leaking managed memory.
    """
    H, W = host_np.shape
    n_bytes = H * W * np.dtype(np.float32).itemsize

    # Create managed memory resource for unified memory allocation
    options = ManagedMemoryResourceOptions(preferred_location=device.device_id)
    mr = ManagedMemoryResource(options)

    # Allocate unified memory buffers for source and destination images
    src_buf = mr.allocate(n_bytes, stream)
    dst_buf = mr.allocate(n_bytes, stream)
    try:
        # Synchronize to ensure allocations are complete before CPU access
        stream.sync()

        # Create numpy views of unified memory using DLPack protocol (zero-copy)
        src_np = np.from_dlpack(src_buf).view(np.float32).reshape(H, W)
        dst_np = np.from_dlpack(dst_buf).view(np.float32).reshape(H, W)

        # Write input data to unified memory (CPU can access directly)
        src_np[:] = host_np.astype(np.float32) / 255.0

        # Configure kernel launch parameters
        block_size = (16, 16)
        grid_size = (
            (W + block_size[0] - 1) // block_size[0],
            (H + block_size[1] - 1) // block_size[1],
        )

        # Create LaunchConfig for kernel execution
        config = LaunchConfig(grid=grid_size, block=block_size)

        # Launch kernel - buffers can be passed directly as kernel arguments
        launch(
            stream,
            config,
            kernel,
            src_buf,
            dst_buf,
            np.int32(H),
            np.int32(W),
        )

        # Synchronize to ensure kernel completion before reading results
        stream.sync()

        # Return zero-copy view; caller closes buffers when done
        return (dst_np, src_buf, dst_buf)
    except Exception:
        src_buf.close()
        dst_buf.close()
        raise


def main():
    """
    Complete demonstration of GPU image blurring with cuda.core.

    This example shows:
    1. Device initialization with cuda.core.Device
    2. Kernel compilation with cuda.core.Program
    3. Unified memory with cuda.core.ManagedMemoryResource
    4. Kernel launch with cuda.core.launch and LaunchConfig
    """
    print("=" * 60)
    print("Image Blur with Unified Memory (cuda.core)")
    print("=" * 60)

    # Initialize CUDA device
    device = Device(0)
    device.set_current()

    print(f"\nDevice: {device.name}")
    print(f"Compute Capability: sm_{device.arch}")

    # Create stream for async operations
    stream = device.create_stream()
    try:
        # Compile kernel using cuda.core.Program
        print("\nCompiling CUDA kernel with cuda.core.Program...")
        arch = f"sm_{device.arch}"
        options = ProgramOptions(arch=arch)
        program = Program(BOX_BLUR_KERNEL_CODE, code_type="c++", options=options)
        compiled = program.compile(target_type="cubin")
        kernel = compiled.get_kernel("box_blur_3x3")
        print(f"  Compiled for architecture: {arch}")

        # Image parameters
        H, W = 256, 256
        print(f"\nImage size: {H}x{W} grayscale")

        # Create test image
        print("Creating sample image...")
        host_np = make_test_image(H, W, dtype=np.uint8)

        # Blur image on GPU using cuda.core (returns zero-copy view + buffers)
        print("Blurring image on GPU...")
        blurred_result, src_buf, dst_buf = blur_image_unified_memory(
            host_np, device, stream, kernel
        )
        try:
            # Save images (use zero-copy view before releasing buffers)
            print("\nSaving results...")
            original_pil = Image.fromarray(host_np, mode="L")
            original_pil.save("original_image.png")
            print("  Saved: original_image.png")

            blurred_uint8 = (np.clip(blurred_result, 0, 1) * 255).astype(np.uint8)
            blurred_pil = Image.fromarray(blurred_uint8, mode="L")
            blurred_pil.save("blurred_image.png")
            print("  Saved: blurred_image.png")

            # Verify blur was applied
            print("\nVerifying result...")
            original_float = host_np.astype(np.float32) / 255.0
            max_diff = np.max(np.abs(blurred_result - original_float))
            blur_applied = max_diff > 0.01

            if blur_applied:
                print("  Test PASSED")
            else:
                print("  Test FAILED - blur not applied")
                sys.exit(1)

            print(f"  Max difference from original: {max_diff:.4f}")
        finally:
            src_buf.close()
            dst_buf.close()
    finally:
        stream.close()


if __name__ == "__main__":
    main()
