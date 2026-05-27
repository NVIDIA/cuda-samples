# Sample: Device Query (Python)

## Description

Query and display detailed properties of all CUDA-capable devices in your system using the modern `cuda.core` API.

## What You'll Learn

- How to enumerate CUDA devices in the system
- Using the `cuda.core` API for device management
- Querying comprehensive device properties (compute capability, memory, limits)
- Accessing low-level device attributes via `cuda.bindings`
- Checking peer-to-peer (P2P) access capabilities between GPUs

## Key Libraries

- `cuda.core` - Modern CUDA Python API
- `cuda.bindings` - Low-level CUDA bindings for runtime and driver APIs

## Key APIs

### From `cuda.core`:

- `Device.get_all_devices()` - Get tuple of all available Device instances
- `Device(device_id)` - Get Device object for specific device ID
- `system.get_driver_version()` - Query CUDA driver version
- `Device.set_current()` - Set the current device for API calls
- `Device.properties` - Access comprehensive device properties
- `Device.name` - Get device name string
- `Device.can_access_peer()` - Check P2P access to peer device

### From `cuda.bindings.runtime`:

- `cudart.cudaRuntimeGetVersion()` - Get CUDA runtime version
- `cudart.cudaDeviceGetAttribute()` - Query specific device attributes

### From `cuda.bindings.driver`:

- `cuda.cuMemGetInfo()` - Get memory information for current device

## Device Properties Queried

### Compute Capabilities:
- Compute capability version (major.minor)
- Driver and runtime versions
- Number of multiprocessors and CUDA cores

### Memory Information:
- Total global memory
- Memory clock rate and bus width
- L2 cache size
- Constant and shared memory sizes
- Maximum memory pitch

### Execution Configuration Limits:
- Maximum threads per block and per multiprocessor
- Maximum block dimensions (x, y, z)
- Maximum grid dimensions (x, y, z)
- Warp size
- Registers per block

### Texture Capabilities:
- Maximum texture dimensions (1D, 2D, 3D)
- Maximum layered texture sizes

### Feature Support:
- Unified Addressing (UVA)
- Managed Memory
- Compute Preemption
- Cooperative Kernel Launch
- ECC support
- Host page-locked memory mapping
- Concurrent copy and kernel execution

### System Information:
- PCI bus information
- Compute mode
- Driver mode (Windows only)
- P2P access matrix (multi-GPU systems)

## Requirements

### Hardware:

- NVIDIA GPU with CUDA support (any compute capability)
- No specific GPU memory requirement (query only)

### Software:

- CUDA Toolkit 13.0 or newer (recommended; matches `cuda-python` 13.x)
- Python 3.10 or newer
- `cuda-python` package (>=13.0.0)
- `cuda-core` package (>=0.6.0)

## Installation

Install the required packages from requirements.txt:

```bash
cd cuda-samples/python/1_GettingStarted/deviceQuery
pip install -r requirements.txt
```

The requirements.txt installs:
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=0.6.0)

## How to Run

### Basic usage:

```bash
cd cuda-samples/python/1_GettingStarted/deviceQuery
python deviceQuery.py
```

### Skip P2P information:

```bash
python deviceQuery.py --no-p2p
```

## Expected Output

```
[CUDA Device Query using CUDA Core API]
Detected 1 CUDA Capable device(s)

Device 0: <Your GPU Name>
  CUDA Driver Version / Runtime Version          12.4 / 12.6
  CUDA Capability Major/Minor version number:    8.9
  Total amount of global memory:                 24217 MBytes (25393954816 bytes)
  (132) Multiprocessors, (128) CUDA Cores/MP:    16896 CUDA Cores
  GPU Max Clock rate:                            1980 MHz (1.98 GHz)
  Memory Clock rate:                             10501 Mhz
  Memory Bus Width:                              384-bit
  L2 Cache Size:                                 67108864 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z):  (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z):  (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use cudaSetDevice() with device simultaneously) >

Done
```

**Note:** Output will vary based on your specific GPU model and system configuration.

For multi-GPU systems, the output will include information for all detected devices and a P2P access matrix showing which GPUs can directly access each other's memory.

## Files

- `deviceQuery.py` - Python implementation using cuda.core API
- `requirements.txt` - Sample dependencies

## Use Cases

- **System Diagnostics** - Verify CUDA installation and GPU detection
- **Hardware Profiling** - Understand GPU capabilities before optimization
- **Multi-GPU Systems** - Identify P2P topology for optimal data placement
- **Kernel Development** - Determine execution configuration limits
- **Compatibility Checks** - Verify compute capability requirements

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [cuda.core API Guide](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [CUDA Programming Guide - Device Information](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-enumeration)
