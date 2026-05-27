# Sample: simpleP2P (Python)

## Description

This sample demonstrates peer-to-peer (P2P) memory access between multiple GPUs in CUDA using the cuda.core Python library. P2P allows GPUs to directly access each other's memory without routing data through the host (CPU), enabling efficient multi-GPU applications. This sample detects P2P-capable GPUs, enables peer access, measures bandwidth using CUDA events for accurate GPU-side timing, and launches kernels (using grid-stride loops) that read from one GPU's memory and write to another GPU's memory.

## What you will learn

- How to detect multiple CUDA-capable GPUs using `system.get_num_devices()` and `Device(id)`
- How to check P2P capability between GPU pairs using `device.can_access_peer()`
- How to enable and disable peer access using `DeviceMemoryResource.peer_accessible_by`
- How to allocate device memory on specific GPUs using `DeviceMemoryResource`
- How to perform direct GPU-to-GPU memory transfers with explicit event-based synchronization
- How to measure P2P bandwidth using CUDA events for accurate GPU-side timing
- How to use event-based synchronization between streams for sequential bandwidth measurement
- How to launch kernels on one GPU that access memory from another GPU
- How to compile and launch CUDA kernels using cuda.core's `Program` and `launch` APIs with grid-stride loops
- How to validate multi-GPU computation results
- How to properly clean up resources using try/finally blocks

## Key libraries

- `numpy` - CPU array operations and data initialization
- `cuda-core` - Modern Python interface to CUDA runtime with full P2P support

## Key APIs

**From cuda.core:**
- `system` – Pre-instantiated singleton for system-level CUDA information
- `system.get_num_devices()` – Get number of CUDA-capable devices
- `Device(id)` – Get specific CUDA device handle
- `device.can_access_peer(peer)` – Check if this device can access peer device memory
- `device.set_current()` – Set active device for subsequent operations
- `device.create_stream()` – Create CUDA stream for kernel execution
- `DeviceMemoryResource(device)` – Create memory resource for specific GPU
- `memory_resource.peer_accessible_by` – Get/set which devices can access this memory pool's allocations
  - Example: `mr.peer_accessible_by = [1]` grants device 1 access
  - Example: `mr.peer_accessible_by = []` revokes all access
- `PinnedMemoryResource()` – Allocate pinned (page-locked) host memory
- `EventOptions(enable_timing=True)` – Create options for CUDA events with timing enabled
- `stream.record(options=event_options)` – Record a CUDA event on a stream
- `event.elapsed_time(start_event)` – Get elapsed time in milliseconds between two events
- `stream.wait_event(event)` – Make a stream wait for an event to complete
- `stream.close()` – Clean up stream resources
- `Program()` – Compile CUDA C++ kernel code
- `LaunchConfig()` – Configure kernel launch parameters (grid, block)
- `launch()` – Launch compiled kernel with arguments
- `buffer.copy_from(src, stream=stream)` – Copy data from source buffer asynchronously
- `buffer.copy_to(dst, stream=stream)` – Copy data to destination buffer asynchronously

**From DLPack:**
- `numpy.from_dlpack()` – Create NumPy array view of memory buffer

**Memory Management:**
- Resources (streams, buffers) should be cleaned up using try/finally blocks to ensure proper cleanup even if errors occur
- Streams should be explicitly closed with `stream.close()` in finally blocks

## Peer-to-Peer (P2P): When to Use

### Benefits
- **Direct GPU-to-GPU transfers**: Bypass host memory for faster communication
- **Higher bandwidth**: PCIe or NVLink bandwidth between GPUs (up to 600 GB/s with NVLink)
- **Lower latency**: No CPU involvement in data transfers
- **Efficient multi-GPU**: Essential for scaling deep learning, HPC, and simulation workloads
- **Simplified programming**: Kernels can directly access remote GPU memory

### Requirements
- **Two or more GPUs**: System must have multiple CUDA-capable GPUs
- **P2P support**: GPUs must be P2P-capable (check with `can_access_peer()`)
- **PCIe topology**: Usually requires GPUs on same PCIe root complex
- **Platform support**: Not available on Mac OSX, limited on ARM platforms

### Best Use Cases
1. Multi-GPU deep learning training (model parallelism, data parallelism)
2. Large-scale scientific simulations across multiple GPUs
3. Real-time rendering with multiple GPUs
4. GPU clusters with direct GPU communication
5. Reducing CPU-GPU traffic in multi-GPU systems

## Requirements

1. **Two or more NVIDIA Graphics Cards** with CUDA support and P2P capability
2. **CUDA Drivers** installed on your system
3. **CUDA Toolkit 13.0+** installed on your system
4. **Python 3.10 or newer**
5. **Proper PCIe topology** (GPUs should be on same PCIe root complex for best performance)

**Note**: This sample will gracefully exit if fewer than 2 GPUs are detected or if P2P is not supported between any GPU pair.

**Install packages:**
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy>=2.3.2 cuda-core>=0.6.0 cuda-python>=13.0.0
```

## How to run

Basic usage:
```bash

# Run with default parameters (16M elements = 64MB)
python simpleP2P.py
```

With custom parameters:
```bash
# Use 32M elements (128MB)
python simpleP2P.py --num_elements 33554432

# Show help
python simpleP2P.py --help
```

### Command line arguments

- `--num_elements`: Number of elements in arrays (default: 16777216)
  - Each array uses `num_elements * 4 bytes` (float32)
  - Default: 64 MB per array
  - Sample allocates 2 device buffers + 1 host buffer

## Expected Output

```
======================================================================
simpleP2P - CUDA Python Sample
======================================================================

Starting...

Checking for multiple GPUs...
CUDA-capable device count: 2

Checking GPU(s) for support of peer to peer memory access...
> Peer access from Tesla T10 (GPU0) -> Tesla T10 (GPU1): Yes
> Peer access from Tesla T10 (GPU1) -> Tesla T10 (GPU0): Yes

Using GPU0 (Tesla T10) and GPU1 (Tesla T10)

Allocating buffers (64MB on GPU0, GPU1 and CPU Host)...
  Peer access enabled: GPU0 <-> GPU1
  Peer access status: MR0 accessible by (1,), MR1 accessible by (0,)
  Memory allocated successfully

Measuring P2P bandwidth...
  Performing 100 ping-pong copies between GPUs...
  P2P bandwidth: 12.37 GB/s

Preparing host buffer and memcpy to GPU0...
  Data initialized and copied to GPU

Compiling CUDA kernel...
  Kernels compiled successfully

Run kernel on GPU1, taking source data from GPU0 and writing to GPU1...
  Kernel execution complete

Run kernel on GPU0, taking source data from GPU1 and writing to GPU0...
  Kernel execution complete

Copy data back to host from GPU0 and verify results...

Checking results...
  Comparing 16,777,216 elements...
Test PASSED
  [PASS] Validation PASSED

Disabling peer access...
  Peer access revoked: MR0 accessible by (), MR1 accessible by ()

======================================================================
simpleP2P completed successfully!
======================================================================

Shutting down...
```

**Note**: P2P bandwidth varies based on:
- PCIe generation
- NVLink
- System topology and configuration

## Files

- `simpleP2P.py` – Main Python implementation
- `README.md` – This file
- `requirements.txt` – Python package dependencies
