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
IPC Memory Pool with cuda.core

Share GPU memory between Python processes using CUDA Inter-Process
Communication (IPC) and cuda.core's IPC-enabled memory pools. By default
each worker process has its own CUDA virtual address space and cannot see
allocations made by another process. With an IPC-enabled
``DeviceMemoryResource`` the parent can allocate once, and the child
process can map that same physical GPU memory into its own address space
so both read and write the same bytes.

The sample does a round-trip test:

  1. Parent creates an IPC-enabled ``DeviceMemoryResource`` and allocates
     a ``Buffer``.
  2. Parent fills the buffer with a known pattern.
  3. Parent sends the ``Buffer`` to a child process through an
     ``mp.Queue`` - cuda.core's pickle reducers take care of re-creating
     the memory resource and mapping the buffer in the child.
  4. Child verifies the parent's pattern, writes a new pattern, and
     signals completion.
  5. Parent verifies the child's writes.

IPC requires Linux (POSIX file-descriptor handles) and device support for
memory pools. On unsupported platforms the sample prints a diagnostic and
exits cleanly.
"""

import multiprocessing as mp
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))

try:
    import cupy as cp
    import numpy as np
    from cuda.core import (
        Device,
        DeviceMemoryResource,
        DeviceMemoryResourceOptions,
    )
    from cuda_samples_utils import print_gpu_info  # noqa: E402
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


CHILD_TIMEOUT_SEC = 30


def check_ipc_support(device) -> bool:
    """Return True if this device/platform supports CUDA IPC memory pools."""
    if platform.system() != "Linux":
        print(
            f"IPC via POSIX file descriptors is only supported on Linux "
            f"(detected {platform.system()})."
        )
        return False
    if not device.properties.memory_pools_supported:
        print("Device does not support CUDA memory pools.")
        return False
    if not device.properties.handle_type_posix_file_descriptor_supported:
        print("Device/platform does not support POSIX-fd IPC handles.")
        return False
    return True


def child_worker(q_in, q_out, n_elements, parent_seed, child_seed):
    """Runs in a separate process. Verifies and modifies the shared buffer."""
    device = Device(0)
    device.set_current()
    pid = mp.current_process().pid

    # The Buffer (and its MR) are reconstructed and mapped in this process
    # when the queued object is unpickled. Both ``is_mapped`` flags are
    # True here.
    buffer = q_in.get(timeout=CHILD_TIMEOUT_SEC)
    print(
        f"[child pid={pid}] received buffer: is_mapped={buffer.is_mapped}, "
        f"size={buffer.size}"
    )

    # Build a zero-copy CuPy view of the shared device memory.
    arr = cp.from_dlpack(buffer).view(dtype=cp.float32)

    # Verify the parent's pattern.
    expected_parent = cp.arange(n_elements, dtype=cp.float32) + float(parent_seed)
    if not cp.allclose(arr, expected_parent):
        print("[child] ERROR: parent's pattern did not match expectation")
        buffer.close()
        q_out.put("fail")
        return

    # Write a new pattern for the parent to verify.
    arr[:] = cp.arange(n_elements, dtype=cp.float32) * float(child_seed)
    device.sync()

    buffer.close()
    q_out.put("done")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Share a GPU buffer between two processes via CUDA IPC"
    )
    parser.add_argument(
        "--elements",
        type=int,
        default=1024,
        help="Number of float32 elements in the shared buffer (default: 1024)",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    args = parser.parse_args()

    # CUDA is incompatible with the ``fork`` start method because forked
    # children inherit a corrupt CUDA state. Always use ``spawn``.
    mp.set_start_method("spawn", force=True)

    device = Device(args.device)
    device.set_current()
    print_gpu_info(device)

    if not check_ipc_support(device):
        print("\nCUDA IPC is not available on this system; exiting cleanly.")
        return 0

    N = args.elements
    nbytes = N * np.dtype(np.float32).itemsize
    parent_seed = 100
    child_seed = -1.0

    # Create an IPC-enabled memory pool. Buffers allocated from this MR
    # are picklable and can be shared across processes.
    mr = DeviceMemoryResource(
        device,
        options=DeviceMemoryResourceOptions(
            max_size=max(nbytes * 4, 1 << 20),
            ipc_enabled=True,
        ),
    )
    print(
        "Created IPC-enabled DeviceMemoryResource "
        f"(is_ipc_enabled={mr.is_ipc_enabled})"
    )

    buffer = mr.allocate(nbytes)
    try:
        # Fill the buffer with a known pattern from the parent side.
        arr = cp.from_dlpack(buffer).view(dtype=cp.float32)
        arr[:] = cp.arange(N, dtype=cp.float32) + float(parent_seed)
        device.sync()
        print(f"Parent wrote pattern (first 5 values): {arr[:5].get()}")

        # Launch the child process and hand the buffer over.
        q_to_child = mp.Queue()
        q_from_child = mp.Queue()
        child = mp.Process(
            target=child_worker,
            args=(q_to_child, q_from_child, N, parent_seed, child_seed),
        )
        child.start()
        q_to_child.put(buffer)
        print(f"Parent sent buffer to child pid={child.pid}; waiting...")

        msg = q_from_child.get(timeout=CHILD_TIMEOUT_SEC)
        child.join(timeout=CHILD_TIMEOUT_SEC)

        if msg != "done" or child.exitcode != 0:
            print(f"Child failed: msg={msg!r}, exitcode={child.exitcode}")
            return 1

        # Verify the child's writes are visible from the parent.
        device.sync()
        got = arr[:5].get()
        expected = (np.arange(N, dtype=np.float32) * child_seed)[:5]
        print(f"Parent sees child's pattern (first 5 values): {got}")
        if np.allclose(got, expected):
            print("IPC round-trip: OK")
            return 0
        print(f"IPC round-trip: FAILED (expected {expected})")
        return 1
    finally:
        buffer.close()
        mr.close()


if __name__ == "__main__":
    sys.exit(main())
