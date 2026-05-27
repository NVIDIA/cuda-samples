# Sample: System Information Query (Python)

## Description

This sample demonstrates how to inspect the CUDA driver, NVML, and every
installed GPU through the
[`cuda.core.system`](https://nvidia.github.io/cuda-python/cuda-core/latest/)
module.

`cuda.core.system` wraps the NVIDIA Management Library (NVML) and can be
imported without CUDA being installed or initialized, so it is useful as a
lightweight pre-flight check before any CUDA context is created. The script
prints driver and NVML versions, the current process name, per-device
metadata (name, compute capability, architecture, memory, PCI info,
temperature, performance state), and, on multi-GPU systems, the topology
and peer-to-peer capabilities between each pair of devices.

## What You'll Learn

- Querying CUDA driver and NVML versions with `cuda.core.system`
- Enumerating GPUs without creating a CUDA context
- Reading per-device metadata exposed by NVML (name, UUID, memory usage,
  temperature, performance state)
- Inspecting GPU-to-GPU topology and peer-to-peer (P2P) capabilities

## Key Libraries

- [`cuda.core.system`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - Python wrapper around NVML

## Key APIs

From `cuda.core.system`:

- `get_driver_version()`, `get_driver_version_full()`, `get_driver_branch()` - CUDA driver version tuple and branch string
- `get_nvml_version()` - NVML library version
- `get_num_devices()` - number of GPUs visible to NVML
- `get_process_name(pid)` - process name for a given PID
- `Device(index=...)` - NVML-backed device handle (no CUDA context required)
  - `name`, `uuid`, `cuda_compute_capability`, `arch`, `brand`
  - `memory_info` (`total`, `used`, `free`)
  - `pci_info` (`domain`, `bus`, `device`, `bus_id`)
  - `temperature.sensor(TemperatureSensors.TEMPERATURE_GPU)`
  - `performance_state`
- `get_topology_common_ancestor(dev0, dev1)` - `GpuTopologyLevel` between two devices
- `get_p2p_status(dev0, dev1, GpuP2PCapsIndex.P2P_CAPS_INDEX_READ)` - peer-access capability between two devices

Import stable symbols from the top-level `cuda.core` package (not `cuda.core.experimental`).

## Requirements

1. **NVIDIA Graphics Card** with CUDA support
2. **CUDA Drivers** installed on your system
3. **CUDA Toolkit** installed on your system
4. **Python 3.12 or newer**

### Hardware

- One or more NVIDIA GPUs
- Driver compatible with `cuda-python` 13.x

### Software

- CUDA Toolkit 13.0 or newer (matches `cuda-python` 13.x)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=0.6.0)

## Installation

Install the required packages from `requirements.txt`:

```bash
cd /path/to/cuda-samples/python/1_GettingStarted/systemInfo
pip install -r requirements.txt
```

The `requirements.txt` installs:

- `cuda-python` (>=13.0.0)
- `cuda-core` (>=0.6.0)

## How to Run

### Basic usage

```bash
cd cuda-samples/python/1_GettingStarted/systemInfo
python systemInfo.py
```

### Skip topology queries

Useful on machines with only one GPU or to shorten the output:

```bash
python systemInfo.py --no-topology
```

## Expected Output

Output varies with your hardware. On a machine with two GPUs you should see
something like:

```
======================================================================
Driver / NVML
======================================================================
CUDA driver version: 13.2
CUDA driver version (full): (13, 2, 0)
NVML version: (13, 595, 58, 3)
Driver branch: r595_88
Current process: /usr/bin/python

======================================================================
Devices detected: 2
======================================================================

-- Device 0 --
Name: <Your GPU Name>
UUID: ...
Compute capability: 8.9
Architecture: ADA
Brand: BRAND_GEFORCE
Memory: total=23.99 GiB, used=960.00 KiB, free=23.52 GiB
PCI: domain=0000 bus=41 device=00 id=00000000:41:00.0
Temperature (GPU sensor): 47 C
Performance state: <Pstates.PSTATE_8: 8>

...

======================================================================
GPU topology and peer-to-peer
======================================================================
Device 0 <-> Device 1: topology=TOPOLOGY_HOSTBRIDGE, p2p_read=..., p2p_write=...

Done
```

**Note:** Device names, compute capability, temperatures, and topology
details will vary based on your GPUs and system.

## Files

- `systemInfo.py` - Python implementation using `cuda.core.system`
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.core.system` API reference](https://nvidia.github.io/cuda-python/cuda-core/latest/api.html)
- [NVML reference](https://docs.nvidia.com/deploy/nvml-api/)
