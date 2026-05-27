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
System Information via cuda.core.system (NVML)

Demonstrates the ``cuda.core.system`` module, which wraps NVIDIA Management
Library (NVML) functionality.

This sample prints:
  * Driver and NVML versions
  * Current process name
  * Per-device: name, UUID, compute capability / arch, PCI info, memory usage,
    temperature, performance state
  * GPU-to-GPU topology and peer-to-peer status (when more than one GPU)
"""

import os
import sys

try:
    from cuda.core import system
    from cuda.core.system import (
        CUDA_BINDINGS_NVML_IS_COMPATIBLE,
        GpuP2PCapsIndex,
        TemperatureSensors,
    )
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def format_bytes(nbytes: int) -> str:
    """Format a byte count as a human-readable string."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(nbytes)
    for unit in units:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PiB"


def print_driver_info() -> None:
    print_header("Driver / NVML")
    major, minor = system.get_driver_version()
    print(f"CUDA driver version: {major}.{minor}")
    print(f"CUDA driver version (full): {system.get_driver_version_full()}")
    if CUDA_BINDINGS_NVML_IS_COMPATIBLE:
        print(f"NVML version: {system.get_nvml_version()}")
        try:
            print(f"Driver branch: {system.get_driver_branch()}")
        except Exception as e:  # noqa: BLE001 - driver branch is informational
            print(f"Driver branch: unavailable ({e})")
    else:
        print(
            "NVML bindings are not compatible with this driver; "
            "device info will be limited."
        )
    print(f"Current process: {system.get_process_name(os.getpid())}")


def print_device_info(device: "system.Device") -> None:
    print(f"\n-- Device {device.index} --")
    print(f"Name: {device.name}")
    print(f"UUID: {device.uuid}")
    try:
        cc_major, cc_minor = device.cuda_compute_capability
        print(f"Compute capability: {cc_major}.{cc_minor}")
    except Exception as e:  # noqa: BLE001
        print(f"Compute capability: unavailable ({e})")
    try:
        print(f"Architecture: {device.arch.name}")
    except Exception as e:  # noqa: BLE001
        print(f"Architecture: unavailable ({e})")
    try:
        print(f"Brand: {device.brand.name}")
    except Exception as e:  # noqa: BLE001
        print(f"Brand: unavailable ({e})")

    # Memory
    try:
        mem = device.memory_info
        print(
            f"Memory: total={format_bytes(mem.total)}, "
            f"used={format_bytes(mem.used)}, "
            f"free={format_bytes(mem.free)}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"Memory: unavailable ({e})")

    # PCI
    try:
        pci = device.pci_info
        print(
            f"PCI: domain={pci.domain:04x} bus={pci.bus:02x} "
            f"device={pci.device:02x} id={pci.bus_id}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"PCI: unavailable ({e})")

    # Temperature (GPU sensor)
    try:
        temp_c = device.temperature.sensor(TemperatureSensors.TEMPERATURE_GPU)
        print(f"Temperature (GPU sensor): {temp_c} C")
    except Exception as e:  # noqa: BLE001
        print(f"Temperature: unavailable ({e})")

    # Performance state
    try:
        pstate = device.performance_state
        print(f"Performance state: {pstate}")
    except Exception as e:  # noqa: BLE001
        print(f"Performance state: unavailable ({e})")


def print_topology(devices: list) -> None:
    if len(devices) < 2:
        return
    print_header("GPU topology and peer-to-peer")
    for i, d0 in enumerate(devices):
        for d1 in devices[i + 1 :]:
            try:
                level = system.get_topology_common_ancestor(d0, d1)
                level_name = level.name
            except Exception as e:  # noqa: BLE001
                level_name = f"unavailable ({e})"
            try:
                read = system.get_p2p_status(
                    d0, d1, GpuP2PCapsIndex.P2P_CAPS_INDEX_READ
                )
                write = system.get_p2p_status(
                    d0, d1, GpuP2PCapsIndex.P2P_CAPS_INDEX_WRITE
                )
                read_name = read.name
                write_name = write.name
            except Exception as e:  # noqa: BLE001
                read_name = write_name = f"unavailable ({e})"
            print(
                f"Device {d0.index} <-> Device {d1.index}: "
                f"topology={level_name}, p2p_read={read_name}, p2p_write={write_name}"
            )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Print CUDA system / NVML information via cuda.core.system"
    )
    parser.add_argument(
        "--no-topology",
        action="store_true",
        help="Skip cross-device topology/P2P queries",
    )
    args = parser.parse_args()

    print_driver_info()

    num_devices = system.get_num_devices()
    print_header(f"Devices detected: {num_devices}")
    if num_devices == 0:
        print("No CUDA-capable devices found.")
        return 0
    if not CUDA_BINDINGS_NVML_IS_COMPATIBLE:
        print(
            "NVML is not compatible with the installed driver; skipping device detail."
        )
        return 0

    devices = [system.Device(index=i) for i in range(num_devices)]
    for device in devices:
        print_device_info(device)

    if not args.no_topology:
        print_topology(devices)

    print("\nDone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
