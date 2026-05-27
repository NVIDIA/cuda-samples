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
Memory management with cuda.core: Buffers and Memory Resources

Demonstrates the Memory Resource / Buffer abstraction in cuda.core:

  * ``DeviceMemoryResource``  - GPU-only memory (device pool)
  * ``PinnedMemoryResource``  - page-locked host memory accessible by the GPU
  * ``ManagedMemoryResource`` - unified memory that migrates between
                                host and device on demand

Each resource hands out ``Buffer`` objects that can be:
  * passed to kernels as pointers
  * copied between each other with ``buffer.copy_to(...)``
  * viewed as NumPy or CuPy arrays via DLPack (``__dlpack__``)

The kernel below performs a fused scale + bias on both a device buffer
and a pinned buffer, then we copy the result across resources to confirm
each pathway works end-to-end.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))

try:
    import cupy as cp
    import numpy as np
    from cuda.core import (
        Device,
        DeviceMemoryResource,
        LaunchConfig,
        ManagedMemoryResource,
        PinnedMemoryResource,
        Program,
        ProgramOptions,
        launch,
    )
    from cuda_samples_utils import print_gpu_info  # noqa: E402
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


SCALE_BIAS_KERNEL = r"""
extern "C" __global__
void scale_and_bias(float* data, size_t N, float scale, float bias) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < N; i += stride) {
        data[i] = data[i] * scale + bias;
    }
}
"""


def demo_device_and_pinned(device, stream, kernel, size):
    """Use pinned host memory as a staging area for a device-side kernel.

    Canonical H2D / compute / D2H pattern:
      host (pinned) -> device -> launch -> device -> host (pinned)
    """
    print("\n[1] DeviceMemoryResource + PinnedMemoryResource (staging)")
    dtype = np.float32
    nbytes = size * dtype().itemsize

    # The device's built-in memory resource is a good default for GPU memory.
    device_mr = device.memory_resource
    pinned_mr = PinnedMemoryResource()

    pinned_in = pinned_mr.allocate(nbytes, stream=stream)
    pinned_out = pinned_mr.allocate(nbytes, stream=stream)
    device_buffer = device_mr.allocate(nbytes, stream=stream)
    try:
        # Wrap each Buffer as a typed array via DLPack (no copies).
        pinned_in_view = np.from_dlpack(pinned_in).view(dtype=dtype)
        pinned_out_view = np.from_dlpack(pinned_out).view(dtype=dtype)

        # Initialize host-side input.
        pinned_in_view[:] = np.arange(size, dtype=dtype)
        original = pinned_in_view.copy()

        # Stage H2D: pinned -> device.
        pinned_in.copy_to(device_buffer, stream=stream)

        # Launch kernel on the device buffer.
        config = LaunchConfig(grid=(size + 255) // 256, block=256)
        launch(
            stream,
            config,
            kernel,
            device_buffer,
            np.uint64(size),
            np.float32(3.0),
            np.float32(-0.5),
        )

        # Stage D2H: device -> pinned.
        device_buffer.copy_to(pinned_out, stream=stream)
        stream.sync()

        expected = original * 3.0 - 0.5
        assert np.allclose(pinned_out_view, expected), "H2D -> kernel -> D2H mismatch"
        print("  Pinned staging, device kernel, and copy_to verified")
    finally:
        device_buffer.close(stream)
        pinned_out.close(stream)
        pinned_in.close(stream)


def demo_managed(device, stream, kernel, size):
    """Allocate a managed (unified) buffer; kernel writes are visible on host."""
    print("\n[2] ManagedMemoryResource (unified memory)")
    dtype = np.float32
    nbytes = size * dtype().itemsize

    managed_mr = ManagedMemoryResource()
    managed_buffer = managed_mr.allocate(nbytes, stream=stream)
    try:
        managed_view = np.from_dlpack(managed_buffer).view(dtype=dtype)

        managed_view[:] = np.arange(size, dtype=dtype)
        original = managed_view.copy()
        # Before launching, make sure host writes have reached the GPU.
        device.sync()

        config = LaunchConfig(grid=(size + 255) // 256, block=256)
        launch(
            stream,
            config,
            kernel,
            managed_buffer,
            np.uint64(size),
            np.float32(0.5),
            np.float32(10.0),
        )
        stream.sync()

        # No explicit copy: the same numpy view observes the GPU's writes.
        assert np.allclose(managed_view, original * 0.5 + 10.0), (
            "Managed memory result mismatch"
        )
        print("  GPU writes observed directly through the host-visible mapping")
    finally:
        managed_buffer.close(stream)


def demo_explicit_device_pool(device, stream, kernel, size):
    """Allocate from a user-created DeviceMemoryResource with default options."""
    print("\n[3] Explicit DeviceMemoryResource")
    dtype = np.float32
    nbytes = size * dtype().itemsize

    # Explicitly create a pool tied to this device. Use .close() to tear it down.
    explicit_mr = DeviceMemoryResource(device)
    buffer = explicit_mr.allocate(nbytes, stream=stream)
    try:
        view = cp.from_dlpack(buffer).view(dtype=cp.float32)
        view[:] = cp.arange(size, dtype=cp.float32)
        device.sync()

        config = LaunchConfig(grid=(size + 255) // 256, block=256)
        launch(
            stream,
            config,
            kernel,
            buffer,
            np.uint64(size),
            np.float32(1.0),
            np.float32(100.0),
        )
        stream.sync()

        expected = cp.arange(size, dtype=cp.float32) + 100.0
        assert cp.allclose(view, expected), "Explicit device pool result mismatch"
        print("  Explicit DeviceMemoryResource allocation verified")
    finally:
        buffer.close(stream)
        explicit_mr.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstrate cuda.core memory resources (Buffer + MR)"
    )
    parser.add_argument(
        "--elements",
        type=int,
        default=1 << 16,
        help="Number of float32 elements per buffer (default: 65536)",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    args = parser.parse_args()

    device = Device(args.device)
    device.set_current()
    print_gpu_info(device)

    stream = device.create_stream()

    try:
        program_options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
        program = Program(SCALE_BIAS_KERNEL, code_type="c++", options=program_options)
        module = program.compile("cubin")
        kernel = module.get_kernel("scale_and_bias")

        demo_device_and_pinned(device, stream, kernel, args.elements)
        demo_managed(device, stream, kernel, args.elements)
        demo_explicit_device_pool(device, stream, kernel, args.elements)

        print("\nDone")
        return 0
    finally:
        stream.close()


if __name__ == "__main__":
    sys.exit(main())
