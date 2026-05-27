# ipcMemoryPool (Python)

## Description

This sample demonstrates how to share GPU memory between Python
processes using CUDA Inter-Process Communication (IPC) and
`cuda.core`'s IPC-enabled memory pools.

By default each process has its own CUDA virtual address space and
cannot see allocations made by another process. With an IPC-enabled
`DeviceMemoryResource` the parent allocates once, and the child
process maps that same physical GPU memory into its own address space
so both read and write the same bytes. The sample performs a
round-trip test:

1. Parent creates an IPC-enabled `DeviceMemoryResource` and allocates
   a `Buffer`.
2. Parent fills the buffer with a known pattern.
3. Parent sends the `Buffer` to a child process through an
   `multiprocessing.Queue`. cuda.core's pickle reducers re-create the
   memory resource and map the buffer in the child.
4. Child verifies the parent's pattern, writes a new pattern, and
   signals completion.
5. Parent verifies the child's writes.

## What You'll Learn

- Enabling IPC on a `DeviceMemoryResource` with `ipc_enabled=True`
- Sending `Buffer` objects across process boundaries via `mp.Queue`
- How cuda.core's pickle reducers rebuild the MR and map the buffer
  in the receiving process
- Why `multiprocessing` must use the `"spawn"` start method with CUDA
- Detecting IPC support at runtime (POSIX file-descriptor handle
  type, memory-pool support, Linux-only)

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - IPC-enabled memory resources and buffer reducers
- `cupy` - zero-copy views over the shared device memory via DLPack
- `multiprocessing` - standard library process management

## Key APIs

### From `cuda.core`

- `DeviceMemoryResource(device, options=DeviceMemoryResourceOptions(ipc_enabled=True))` - create an IPC-enabled memory pool
- `DeviceMemoryResourceOptions(max_size=..., ipc_enabled=True)` - configure the underlying pool
- `mr.allocate(nbytes)` - allocate a `Buffer` from the IPC pool
- `Buffer.is_mapped` - True when the buffer is usable in the current process
- `Device.properties.memory_pools_supported` - runtime feature check
- `Device.properties.handle_type_posix_file_descriptor_supported` - runtime feature check

### From `cuda_samples_utils`

- `print_gpu_info()` - print device name and compute capability

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.0 or higher
- Device that supports CUDA memory pools and POSIX file-descriptor IPC handles (the sample detects and reports this at startup)
- Minimum GPU memory: 512 MB

### Software

- Linux x86_64 (POSIX file-descriptor IPC handles are not available on Windows or macOS)
- CUDA Toolkit 13.0 or newer (matches `cuda-python` 13.x)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=0.6.0)
- `cupy-cuda13x` (>=13.0.0)

## Installation

Install the required packages from `requirements.txt`:

```bash
cd /path/to/cuda-samples/python/4_DistributedComputing/ipcMemoryPool
pip install -r requirements.txt
```

The `requirements.txt` installs:

- `cuda-python` (>=13.0.0)
- `cuda-core` (>=0.6.0)
- `cupy-cuda13x` (>=13.0.0)

## How to Run

### Basic usage

```bash
cd cuda-samples/python/4_DistributedComputing/ipcMemoryPool
python ipcMemoryPool.py
```

### With custom parameters

```bash
# Larger shared buffer
python ipcMemoryPool.py --elements 65536

# Use a specific GPU
python ipcMemoryPool.py --device 1
```

On platforms or devices that do not support CUDA IPC, the sample
prints a diagnostic and exits cleanly with status 0.

## Expected Output

```
Device: <Your GPU Name>
Compute Capability: <X.Y>

Created IPC-enabled DeviceMemoryResource (is_ipc_enabled=True)
Parent wrote pattern (first 5 values): [100. 101. 102. 103. 104.]
Parent sent buffer to child pid=<pid>; waiting...
[child pid=<pid>] received buffer: is_mapped=True, size=4096
Parent sees child's pattern (first 5 values): [-0. -1. -2. -3. -4.]
IPC round-trip: OK
```

**Note:** Device name, compute capability, and child PID will vary
based on your system.

## Files

- `ipcMemoryPool.py` - Python implementation using `cuda.core` IPC memory pools
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_samples_utils.py` - Common utilities (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.core` memory API](https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#memory)
- Upstream `cuda.core` IPC tests: [`test_memory_ipc.py`](https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/tests/memory_ipc/test_memory_ipc.py)
- [CUDA IPC programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#interprocess-communication)
