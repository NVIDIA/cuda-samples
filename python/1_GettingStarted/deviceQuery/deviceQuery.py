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
Device Query using CUDA Core API

This sample enumerates the properties of the CUDA devices present in the system.
"""

import platform
import sys

# cuda.bindings used for properties not yet exposed in cuda.core (see comments below)
try:
    from cuda.bindings import driver as cuda, runtime as cudart
    from cuda.core import Device, system
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def print_property(label, value, indent=2):
    """
    Helper function to print device properties with aligned formatting.

    Parameters
    ----------
    label : str
        Property label
    value : any
        Property value
    indent : int
        Number of spaces for indentation (default: 2)
    """
    field_width = 47
    spaces = " " * indent
    print(f"{spaces}{label:<{field_width}}{value}")


def fmt_bytes(size_in_bytes):
    """Format bytes to human-readable string with MBytes."""
    return f"{size_in_bytes / (1024 * 1024):.0f} MBytes ({size_in_bytes} bytes)"


def fmt_hz(rate_in_khz):
    """Format frequency in kHz to MHz and GHz."""
    return f"{rate_in_khz * 1e-3:.0f} MHz ({rate_in_khz * 1e-6:.2f} GHz)"


def fmt_yes_no(val):
    """Format boolean value to Yes/No string."""
    return "Yes" if val else "No"


def convert_sm_ver_to_cores(major, minor):
    """
    Maps SM version to the number of CUDA cores per SM.

    Information taken from:
    https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h

    Parameters
    ----------
    major : int
        Major compute capability version
    minor : int
        Minor compute capability version

    Returns
    -------
    int
        Number of CUDA cores per SM, or 0 if unknown
    """
    sm_to_cores = {
        (3, 0): 192,
        (3, 2): 192,
        (3, 5): 192,
        (3, 7): 192,
        (5, 0): 128,
        (5, 2): 128,
        (5, 3): 128,
        (6, 0): 64,
        (6, 1): 128,
        (6, 2): 128,
        (7, 0): 64,
        (7, 2): 64,
        (7, 5): 64,
        (8, 0): 64,
        (8, 6): 128,
        (8, 7): 128,
        (8, 9): 128,
        (9, 0): 128,
        (10, 0): 128,
        (10, 1): 128,
        (10, 3): 128,
        (11, 0): 128,
        (12, 0): 128,
        (12, 1): 128,
    }
    return sm_to_cores.get((major, minor), 0)


def print_device_info(dev_id, device):
    """
    Print detailed information for a single CUDA device.
    Uses device.properties (cuda.core) for most fields; cuda.bindings for
    runtime version and global memory (not yet in high-level API).
    """
    device.set_current()
    props = device.properties

    print()
    print(f"Device {dev_id}: {device.name}")

    # cuda.bindings workaround: runtime version not in cuda.core
    driver_major, driver_minor = system.get_driver_version()
    err, runtime_version = cudart.cudaRuntimeGetVersion()
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Failed to get CUDA runtime version: {err}")
    runtime_major = runtime_version // 1000
    runtime_minor = (runtime_version % 1000) // 10

    print_property(
        "CUDA Driver Version / Runtime Version",
        f"{driver_major}.{driver_minor} / {runtime_major}.{runtime_minor}",
    )
    print_property(
        "CUDA Capability Major/Minor version number:",
        f"{props.compute_capability_major}.{props.compute_capability_minor}",
    )

    # cuda.bindings workaround: global memory (free/total) not in device.properties
    err, free_mem, total_mem_bytes = cuda.cuMemGetInfo()
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to get memory info: {err}")
    print_property("Total amount of global memory:", fmt_bytes(total_mem_bytes))

    sm_cores = convert_sm_ver_to_cores(
        props.compute_capability_major, props.compute_capability_minor
    )
    total_cores = sm_cores * props.multiprocessor_count
    print_property(
        f"({props.multiprocessor_count:3d}) Multiprocessors, "
        f"({sm_cores:3d}) CUDA Cores/MP:",
        f"{total_cores} CUDA Cores",
    )

    print_property("GPU Max Clock rate:", fmt_hz(props.clock_rate))
    print_property("Memory Clock rate:", f"{props.memory_clock_rate * 1e-3:.0f} Mhz")
    print_property("Memory Bus Width:", f"{props.global_memory_bus_width}-bit")
    if props.l2_cache_size > 0:
        print_property("L2 Cache Size:", f"{props.l2_cache_size} bytes")

    print_property(
        "Maximum Texture Dimension Size (x,y,z)",
        f"1D=({props.maximum_texture1d_width}), "
        f"2D=({props.maximum_texture2d_width}, {props.maximum_texture2d_height}), "
        f"3D=({props.maximum_texture3d_width}, {props.maximum_texture3d_height}, "
        f"{props.maximum_texture3d_depth})",
    )
    print_property(
        "Maximum Layered 1D Texture Size, (num) layers",
        f"1D=({props.maximum_texture1d_layered_width}), "
        f"{props.maximum_texture1d_layered_layers} layers",
    )
    print_property(
        "Maximum Layered 2D Texture Size, (num) layers",
        f"2D=({props.maximum_texture2d_layered_width}, "
        f"{props.maximum_texture2d_layered_height}), "
        f"{props.maximum_texture2d_layered_layers} layers",
    )

    print_property(
        "Total amount of constant memory:", f"{props.total_constant_memory} bytes"
    )
    print_property(
        "Total amount of shared memory per block:",
        f"{props.max_shared_memory_per_block} bytes",
    )
    print_property(
        "Total shared memory per multiprocessor:",
        f"{props.max_shared_memory_per_multiprocessor} bytes",
    )
    print_property(
        "Total number of registers available per block:", props.max_registers_per_block
    )

    print_property("Warp size:", props.warp_size)
    print_property(
        "Maximum number of threads per multiprocessor:",
        props.max_threads_per_multiprocessor,
    )
    print_property("Maximum number of threads per block:", props.max_threads_per_block)
    print_property(
        "Max dimension size of a thread block (x,y,z):",
        f"({props.max_block_dim_x}, {props.max_block_dim_y}, {props.max_block_dim_z})",
    )
    print_property(
        "Max dimension size of a grid size    (x,y,z):",
        f"({props.max_grid_dim_x}, {props.max_grid_dim_y}, {props.max_grid_dim_z})",
    )
    print_property("Maximum memory pitch:", f"{props.max_pitch} bytes")
    print_property("Texture alignment:", f"{props.texture_alignment} bytes")

    print_property(
        "Concurrent copy and kernel execution:",
        f"{fmt_yes_no(props.gpu_overlap)} with "
        f"{props.async_engine_count} copy engine(s)",
    )
    print_property("Run time limit on kernels:", fmt_yes_no(props.kernel_exec_timeout))

    print_property("Integrated GPU sharing Host Memory:", fmt_yes_no(props.integrated))
    print_property(
        "Support host page-locked memory mapping:",
        fmt_yes_no(props.can_map_host_memory),
    )
    print_property(
        "Device has ECC support:", "Enabled" if props.ecc_enabled else "Disabled"
    )
    if platform.system() == "Windows":
        mode = (
            "TCC (Tesla Compute Cluster Driver)"
            if props.tcc_driver
            else "WDDM (Windows Display Driver Model)"
        )
        print_property("CUDA Device Driver Mode (TCC or WDDM):", mode)

    print_property(
        "Device supports Unified Addressing (UVA):",
        fmt_yes_no(props.unified_addressing),
    )
    print_property("Device supports Managed Memory:", fmt_yes_no(props.managed_memory))
    print_property(
        "Device supports Compute Preemption:",
        fmt_yes_no(props.compute_preemption_supported),
    )
    print_property(
        "Supports Cooperative Kernel Launch:", fmt_yes_no(props.cooperative_launch)
    )

    print_property(
        "Device PCI Domain ID / Bus ID / location ID:",
        f"{props.pci_domain_id} / {props.pci_bus_id} / {props.pci_device_id}",
    )
    compute_modes = {
        0: (
            "Default (multiple host threads can use cudaSetDevice() "
            "with device simultaneously)"
        ),
        1: (
            "Exclusive (only one host thread in one process is able to "
            "use cudaSetDevice() with this device)"
        ),
        2: "Prohibited (no host thread can use cudaSetDevice() with this device)",
        3: (
            "Exclusive Process (many threads in one process is able to "
            "use cudaSetDevice() with this device)"
        ),
    }
    print_property("Compute Mode:", "")
    print(f"     < {compute_modes.get(props.compute_mode, 'Unknown')} >")


def print_p2p_access_info(devices):
    """
    Print peer-to-peer access information for multi-GPU systems.

    Parameters
    ----------
    devices : tuple of Device
        Tuple of CUDA device objects
    """
    print()
    print("Peer-to-Peer (P2P) access support:")
    for i, dev_i in enumerate(devices):
        for j, dev_j in enumerate(devices):
            if i == j:
                continue
            try:
                can_access = dev_i.can_access_peer(dev_j)
                print(
                    f"> Peer access from {dev_i.name} (GPU{i}) -> "
                    f"{dev_j.name} (GPU{j}) : {fmt_yes_no(can_access)}"
                )
            except Exception as e:
                print(
                    "Warning: Could not check peer access between "
                    f"device {i} and {j}: {e}"
                )


def query_devices(show_p2p=True):
    """
    Query and display information about all CUDA devices.

    Parameters
    ----------
    show_p2p : bool
        Whether to show peer-to-peer access information (default: True)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        print("[CUDA Device Query using CUDA Core API]")
        devices = Device.get_all_devices()
    except Exception as e:
        print(f"Error: Failed to get devices: {e}")
        import traceback

        traceback.print_exc()
        return False

    if len(devices) == 0:
        print("There are no available device(s) that support CUDA")
        return True

    print(f"Detected {len(devices)} CUDA Capable device(s)")

    for dev_id, device in enumerate(devices):
        try:
            print_device_info(dev_id, device)
        except Exception as e:
            print(f"Error: Failed to get information for device {dev_id}: {e}")
            import traceback

            traceback.print_exc()
            return False

    if show_p2p and len(devices) >= 2:
        print_p2p_access_info(devices)

    return True


def main():
    """
    Main entry point for the device query sample.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Query CUDA Device Properties using CUDA Core API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-p2p", action="store_true", help="Skip peer-to-peer access information"
    )

    args = parser.parse_args()

    success = query_devices(show_p2p=not args.no_p2p)

    if success:
        print("\nDone")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
