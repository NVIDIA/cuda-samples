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

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    from cuda.core import (
        Device,
        DeviceMemoryResource,
        EventOptions,
        LaunchConfig,
        PinnedMemoryResource,
        Program,
        ProgramOptions,
        launch,
        system,
    )
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Add parent directory to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))
from cuda_samples_utils import verify_array_result  # noqa: E402

# CUDA kernel for simple P2P operation
SIMPLE_P2P_KERNEL = """
extern "C" __global__
void SimpleKernel(float *src, float *dst, int N) {
    // Grid-stride loop pattern for canonical CUDA kernel
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < N; i += stride) {
        dst[i] = src[i] * 2.0f;
    }
}
"""


def run(num_elements=1024 * 1024 * 16):
    """
    Demonstrates peer-to-peer (P2P) memory access between multiple GPUs using cuda.core.

    This function shows how to:
    1. Detect and verify multiple GPUs with P2P capability
    2. Enable peer access between GPUs
    3. Perform direct GPU-to-GPU memory transfers
    4. Launch kernels that access memory from other GPUs
    5. Measure P2P bandwidth
    6. Validate results

    Parameters
    ----------
    num_elements : int
        Number of elements in arrays (default: 16M elements = 64MB)
    """

    print("\n" + "=" * 70)
    print("simpleP2P - CUDA Python Sample")
    print("=" * 70)
    print("\nStarting...")

    # Check for multiple GPUs
    print("\nChecking for multiple GPUs...")
    num_devices = system.get_num_devices()
    print(f"CUDA-capable device count: {num_devices}")

    if num_devices < 2:
        print(
            "Two or more GPUs with Peer-to-Peer access capability are "
            "required, waiving this sample."
        )
        return 2

    # Get device properties
    devices = [Device(i) for i in range(num_devices)]

    # Check for P2P capability
    print("\nChecking GPU(s) for support of peer to peer memory access...")

    p2p_capable_gpus = [-1, -1]

    for i in range(num_devices):
        p2p_capable_gpus[0] = i
        for j in range(num_devices):
            if i == j:
                continue

            # Check peer access capability using cuda.core
            i_access_j = devices[i].can_access_peer(devices[j])
            j_access_i = devices[j].can_access_peer(devices[i])

            print(
                f"> Peer access from {devices[i].name} (GPU{i}) -> "
                f"{devices[j].name} (GPU{j}): {'Yes' if i_access_j else 'No'}"
            )
            print(
                f"> Peer access from {devices[j].name} (GPU{j}) -> "
                f"{devices[i].name} (GPU{i}): {'Yes' if j_access_i else 'No'}"
            )

            if i_access_j and j_access_i:
                p2p_capable_gpus[1] = j
                break

        if p2p_capable_gpus[1] != -1:
            break

    if p2p_capable_gpus[0] == -1 or p2p_capable_gpus[1] == -1:
        print("\nTwo or more GPUs with Peer-to-Peer access capability are required.")
        print(
            "Peer to Peer access is not available amongst GPUs in the system, "
            "waiving test."
        )
        return 2

    # Use first pair of P2P capable GPUs detected
    gpuid = [p2p_capable_gpus[0], p2p_capable_gpus[1]]
    dev0 = devices[gpuid[0]]
    dev1 = devices[gpuid[1]]

    print(f"\nUsing GPU{gpuid[0]} ({dev0.name}) and GPU{gpuid[1]} ({dev1.name})")

    # Allocate buffers with P2P access
    buf_size = num_elements * np.dtype(np.float32).itemsize
    print(
        f"\nAllocating buffers ({int(buf_size / 1024 / 1024)}MB on "
        f"GPU{gpuid[0]}, GPU{gpuid[1]} and CPU Host)..."
    )

    # Allocate on GPU 0 and grant access to GPU 1
    dev0.set_current()
    mr0 = DeviceMemoryResource(dev0)
    mr0.peer_accessible_by = [gpuid[1]]  # Grant GPU 1 access to GPU 0's memory
    g0 = mr0.allocate(buf_size)

    # Allocate on GPU 1 and grant access to GPU 0
    dev1.set_current()
    mr1 = DeviceMemoryResource(dev1)
    mr1.peer_accessible_by = [gpuid[0]]  # Grant GPU 0 access to GPU 1's memory
    g1 = mr1.allocate(buf_size)

    print(f"  Peer access enabled: GPU{gpuid[0]} <-> GPU{gpuid[1]}")
    print(
        f"  Peer access status: MR0 accessible by {mr0.peer_accessible_by}, "
        f"MR1 accessible by {mr1.peer_accessible_by}"
    )

    # Allocate pinned host memory
    pinned_mr = PinnedMemoryResource()
    h0 = pinned_mr.allocate(buf_size)

    print("  Memory allocated successfully")

    # Create streams
    stream0 = dev0.create_stream()
    stream1 = dev1.create_stream()

    try:
        # P2P bandwidth test using CUDA events for accurate GPU-side timing
        print("\nMeasuring P2P bandwidth...")
        print("  Performing 100 ping-pong copies between GPUs...")

        event_options = EventOptions(enable_timing=True)
        sync_event0 = None
        sync_event1 = None

        # Record start event on stream0
        start_event = stream0.record(options=event_options)

        for i in range(100):
            # Ping-pong copy between GPUs with explicit event-based synchronization
            if i % 2 == 0:
                # Wait for previous stream1 copy to complete (if any)
                if sync_event1 is not None:
                    stream0.wait(sync_event1)
                # Copy g0 -> g1 on stream0
                g1.copy_from(g0, stream=stream0)
                # Record event on stream0 to signal completion of this copy
                sync_event0 = stream0.record(options=EventOptions(enable_timing=False))
            else:
                # Wait for previous stream0 copy to complete
                if sync_event0 is not None:
                    stream1.wait(sync_event0)
                # Copy g1 -> g0 on stream1
                g0.copy_from(g1, stream=stream1)
                # Record event on stream1 to signal completion of this copy
                sync_event1 = stream1.record(options=EventOptions(enable_timing=False))

        # Wait for last stream1 copy to complete
        if sync_event1 is not None:
            stream0.wait(sync_event1)

        # Record end event on stream0 after all copies have been enqueued
        end_event = stream0.record(options=event_options)
        end_event.sync()

        # Elapsed time in milliseconds (using subtraction operator)
        time_memcpy = end_event - start_event

        bandwidth = (1.0 / (time_memcpy / 1000.0)) * (100.0 * buf_size) / (1024.0**3)
        print(f"  P2P bandwidth: {bandwidth:.2f} GB/s")

        # Prepare host buffer and initialize data
        print(f"\nPreparing host buffer and memcpy to GPU{gpuid[0]}...")

        # Create numpy view and initialize
        h0_array = np.from_dlpack(h0).view(dtype=np.float32)
        h0_array[:] = (np.arange(num_elements, dtype=np.float32) % 4096).astype(
            np.float32
        )

        # Copy to GPU 0
        dev0.set_current()
        g0.copy_from(h0, stream=stream0)
        stream0.sync()

        print("  Data initialized and copied to GPU")

        # Compile kernel for both GPUs
        print("\nCompiling CUDA kernel...")
        dev0.set_current()
        program_options = ProgramOptions(std="c++17", arch=f"sm_{dev0.arch}")
        prog = Program(SIMPLE_P2P_KERNEL, code_type="c++", options=program_options)
        mod0 = prog.compile("cubin")
        kernel0 = mod0.get_kernel("SimpleKernel")

        dev1.set_current()
        program_options = ProgramOptions(std="c++17", arch=f"sm_{dev1.arch}")
        prog = Program(SIMPLE_P2P_KERNEL, code_type="c++", options=program_options)
        mod1 = prog.compile("cubin")
        kernel1 = mod1.get_kernel("SimpleKernel")

        print("  Kernels compiled successfully")

        # Launch configuration
        threads = 512
        blocks = (num_elements + threads - 1) // threads
        config = LaunchConfig(grid=blocks, block=threads)

        # Run kernel on GPU 1, reading from GPU 0, writing to GPU 1
        print(
            f"\nRun kernel on GPU{gpuid[1]}, taking source data from "
            f"GPU{gpuid[0]} and writing to GPU{gpuid[1]}..."
        )
        dev1.set_current()
        launch(stream1, config, kernel1, g0, g1, np.int32(num_elements))
        stream1.sync()
        print("  Kernel execution complete")

        # Run kernel on GPU 0, reading from GPU 1, writing to GPU 0
        print(
            f"\nRun kernel on GPU{gpuid[0]}, taking source data from "
            f"GPU{gpuid[1]} and writing to GPU{gpuid[0]}..."
        )
        dev0.set_current()
        launch(stream0, config, kernel0, g1, g0, np.int32(num_elements))
        stream0.sync()
        print("  Kernel execution complete")

        # Copy data back to host and verify
        print(f"\nCopy data back to host from GPU{gpuid[0]} and verify results...")
        g0.copy_to(h0, stream=stream0)
        stream0.sync()

        # Verify results
        print("\nChecking results...")
        print(f"  Comparing {num_elements:,} elements...")

        # Input data goes through two kernels, each multiplying by 2.0.
        expected = (np.arange(num_elements, dtype=np.float32) % 4096) * 4.0

        # Use utility function for verification (handles both numpy and cupy arrays)
        if verify_array_result(h0_array, expected, rtol=1e-5, atol=1e-6, verbose=True):
            print("  [PASS] Validation PASSED")
            success = True
        else:
            print("  [FAIL] Validation FAILED")
            # Show first few errors for debugging
            errors = np.where(~np.isclose(h0_array, expected, rtol=1e-5, atol=1e-6))[0]
            print(f"  Number of mismatches: {len(errors)}")
            for idx in errors[:10]:
                print(
                    f"    Error @ element {idx}: got {h0_array[idx]}, "
                    f"expected {expected[idx]}"
                )
            success = False

        # Disable peer access
        print("\nDisabling peer access...")
        mr0.peer_accessible_by = []  # Revoke GPU 1's access to GPU 0's memory
        mr1.peer_accessible_by = []  # Revoke GPU 0's access to GPU 1's memory
        print(
            f"  Peer access revoked: MR0 accessible by {mr0.peer_accessible_by}, "
            f"MR1 accessible by {mr1.peer_accessible_by}"
        )

        print("\n" + "=" * 70)
        if success:
            print("simpleP2P completed successfully!")
        else:
            print("simpleP2P FAILED!")
        print("=" * 70 + "\n")

        return 0 if success else 1
    finally:
        # Cleanup streams and buffers
        print("Shutting down...")
        stream0.close()
        stream1.close()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate peer-to-peer (P2P) memory access between "
            "multiple GPUs with CUDA"
        )
    )

    parser.add_argument(
        "--num_elements",
        type=int,
        default=1024 * 1024 * 16,  # 16M elements = 64MB
        help="Number of elements in arrays (default: 16777216 = 64MB)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.num_elements <= 0:
        print("Error: num_elements must be positive")
        return 1

    try:
        exit_code = run(num_elements=args.num_elements)
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
