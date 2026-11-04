# cftUnicastPut - Unicast Point-to-Point Puts with Compute Fabric Transport

## Description

This sample moves data directly between GPUs with **Compute Fabric Transport (CFT)** unicast *logical endpoints*. A logical endpoint (LE) is a fabric-addressable name for a range of device memory: a rank creates an LE, binds VMM fabric memory to it, shares it with peers through a fabric IPC handle, and peers then write into it from device code with `fabric.try_put`, addressing the destination by `(endpoint id, offset)` rather than by a mapped peer pointer.

The ranks form a ring: rank `r` puts its payload into rank `(r + 1) % N` and expects the payload of rank `(r - 1 + N) % N` to land in its own buffer.

Two receive-completion schemes are shown, selected on the command line:

- **Default (plain puts)** — `cftTryPutKernel` only moves payload. A second, one-thread `cftSignalFlagKernel` then puts a 16-byte flag into the neighbor's sync region and spins on its own flag. Because the two kernels are ordered on the same stream, the flag cannot be observed before the payload puts of that rank have completed.
- **`--counted` (counted puts)** — `cftTryPutCountedKernel` issues `fabric.try_put_counted`, which additionally increments a byte counter in the destination endpoint. The receiver simply waits until its own counter reaches the number of bytes it expects, so no separate signal kernel is needed.

After the transfer each rank copies its buffer back to the host and checks the rank-tagged pattern its receive neighbor sent, and rank 0 prints the combined result.

## What You'll Learn

How to:
- discover the unicast logical-endpoint clique a GPU belongs to and confirm all ranks agree on it
- reserve logical-endpoint ids, create an endpoint, and share it across processes with a fabric IPC handle
- bind VMM fabric memory to an endpoint and tear the endpoint down safely
- issue `fabric.try_put` / `fabric.try_put_counted` from device code and complete them with an `mbarrier`
- decode fabric operation errors from the `mbarrier.try_wait` status operand
- pick a receive-completion scheme: an explicit inter-GPU signal flag put, or a counted put with a byte-arrival counter

## Key Concepts

- **Logical endpoint (LE)** — a fabric-addressable handle for a bound range of device memory; remote writes target `(endpoint id, offset)` instead of a mapped peer address
- **Clique** — the set of devices that can address each other's logical endpoints; every participating rank must find the same `CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT` clique id
- **Fabric put** — an asynchronous device-side write to a peer endpoint issued by a single thread, completed through an `mbarrier` with `complete_tx::16B` accounting
- **Counted put** — a fabric put that also accumulates delivered bytes into a 256-byte-aligned counter in the destination endpoint, giving the receiver a completion signal without an extra message
- **Fabric error reporting** — when the `mbarrier.try_wait` report predicate is set, `cudaFabricOpErrorStatus*` decodes failures from the `try_wait` status operand.

## Key APIs

### CUDA Driver API

- `cuDeviceGetCliqueCount` / `cuDeviceGetCliqueInfo` — find the unicast logical-endpoint clique of a device
- `cuLogicalEndpointIdReserve` / `cuLogicalEndpointIdRelease` — reserve a contiguous range of local endpoint ids
- `cuLogicalEndpointCreate` / `cuLogicalEndpointDestroy` — create and destroy an endpoint
- `cuLogicalEndpointGetLimits` — query the bind alignment and the maximum endpoint size
- `cuLogicalEndpointExport` / `cuLogicalEndpointImport` — share an endpoint through a fabric IPC handle
- `cuLogicalEndpointQuery` — wait until a created or imported endpoint is ready to use
- `cuLogicalEndpointBindMem` / `cuLogicalEndpointUnbind` — attach and detach the backing allocation
- `cuMemCreate`, `cuMemAddressReserve`, `cuMemMap`, `cuMemSetAccess` — allocate the fabric-shareable backing memory

### Device-side PTX wrappers (`cuda::ptx`)

- `fabric_try_put` / `fabric_try_put_counted` — issue the remote write, optionally with byte counting
- `fabric_submit` — submit the batch of issued fabric operations
- `mbarrier_init` (`layout_v1`), `mbarrier_arrive_expect_tx`, `mbarrier_try_wait_parity` — track completion of the puts

### CUDA Runtime API

- `cudaFabricOpErrorStatusCount` / `cudaFabricOpErrorStatusGet` — decode fabric errors from the `mbarrier.try_wait` status field (`cudaFabricOpStatusSourceMbarrierV1`)

## Requirements

### Hardware

- Two or more NVIDIA GPUs of compute capability 10.0 or newer.
- Two or more NVLink-capable datacenter class NVIDIA GPUs of compute capability 10.0 or newer (`sm_100` / `sm_103` / `sm_107`).
- Each rank needs its own device, so a node must expose at least as many GPUs as it runs ranks.

### Software

- CUDA Toolkit 13.4 or newer and a driver with `cuLogicalEndpoint*` support
- The NVIDIA IMEX daemon with IMEX channels configured, as required for any `CU_MEM_HANDLE_TYPE_FABRIC` sharing
- [MPI](../../../README.md#message-passing-interface)
- Linux on x86_64 or aarch64

## How to Build

See the [top-level README](../../../README.md#building-cuda-samples) for full build instructions, including how to build all samples or a single sample standalone.

## How to Run

The sample must be launched under MPI with at least two ranks:

```bash
mpirun -np 2 ./cftUnicastPut [chunkCnt] [--counted]
```

`[chunkCnt]` is optional (default: 1000) and is the number of 8192-byte chunks each rank sends; it also becomes the grid size (one block per chunk). `--counted` switches from plain puts plus a flag handshake to counted puts plus a byte-arrival counter.

```bash
mpirun -np 2 ./cftUnicastPut                 # defaults: 1000 chunks, plain puts
mpirun -np 2 ./cftUnicastPut --counted       # defaults: 1000 chunks, counted puts
mpirun -np 2 ./cftUnicastPut 1000            # plain puts + inter-GPU flag
mpirun -np 2 ./cftUnicastPut 1000 --counted  # counted puts + byte counter
```

Debug builds (`-DENABLE_CUDA_DEBUG=ON`, or any build without `NDEBUG`) cap every device-side spin-wait at 30 seconds and print which wait timed out, so a stalled fabric operation reports itself instead of hanging.

## Expected Output

```text
mode = put, chunkCnt = 1000, totalBytes = 8192000, ranks = 2
Result = PASS
```

If any rank's received payload does not match what its neighbor sent, that rank prints the first mismatching element to `stderr`, rank 0 prints `Result = FAIL`, and the sample exits with a non-zero status.

## Files

- `cft_unicast_put.cu` — the complete sample: endpoint setup and teardown, both put kernels, the flag-signal kernel, and host-side verification
- `README.md` — this file
- `CMakeLists.txt` — build configuration

## See Also

- [CUDA Compute Fabric Transport](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/compute-fabric-transport.html)
- [CUDA Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [`cuda::ptx` PTX API](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx_api.html)
