# Sample: CUDA Graphs (simpleCudaGraphs)

## Description

Demonstrates how to create, instantiate, and launch CUDA Graphs using two different approaches: building the graph explicitly, node by node, with `cudaGraphAddNode`, and building it automatically by recording an existing stream sequence with `cudaStreamBeginCapture` / `cudaStreamEndCapture`. A CUDA Graph records a sequence of GPU operations — memory copies, kernel launches, and host callbacks — as a reusable DAG (Directed Acyclic Graph). Once instantiated, the graph can be submitted to the GPU with a single API call per launch, reducing per-launch CPU overhead compared to issuing each operation individually.

The workload is a two-pass parallel reduction: 16M `float` values are reduced to per-block partial sums (`reduce` kernel), then to a single `double` result (`reduceFinal` kernel). Both samples use `cub::BlockReduce` for the in-kernel reduction. Each graph is launched several times, and the host input is refilled with new random values before every launch, to show that one instantiated graph can reprocess different data each iteration.

This sample is split into two standalone executables plus a shared header:

| File | Role |
|---|---|
| `simpleCudaGraphs_explicit.cu` | Explicit construction via the unified `cudaGraphAddNode` API |
| `simpleCudaGraphs_capture.cu` | Automatic construction via stream capture |
| `simpleCudaGraphs.cuh` | Shared reduction kernels, host callback (and its data type), and input-fill helper used by both |

## What You'll Learn

- Building a CUDA Graph node by node using the unified `cudaGraphAddNode` API with a typed `cudaGraphNodeParams`
- Expressing dependencies between nodes so the runtime enforces the correct execution order
- Building the same graph automatically by recording stream operations between `cudaStreamBeginCapture` and `cudaStreamEndCapture`
- Instantiating a graph with `cudaGraphInstantiate` (one-time compilation cost) and launching it repeatedly with `cudaGraphLaunch`
- Cloning a graph with `cudaGraphClone` to produce independent executable instances
- Adding memcpy, kernel, and host-callback nodes through the single polymorphic `cudaGraphAddNode` entry point
- Reusing one instantiated graph across many launches by refilling its input buffer between launches (and synchronizing so the graph's H2D copy consumes the data before the host overwrites it)
- Using `cub::BlockReduce` for efficient block-level reductions

## Key Concepts

- **CUDA Graph** — a DAG of GPU operations captured once and replayed many times; each replay is a single `cudaGraphLaunch` call regardless of graph size
- **Graph Node** — an individual operation in the graph: memcpy, kernel launch, or host callback
- **Node dependency** — an edge from node A to node B means B cannot start until A completes; expressed as a dependency list passed to `cudaGraphAddNode`
- **Instantiation** — `cudaGraphInstantiate` compiles the graph into an optimized executable form (`cudaGraphExec_t`); this is the one-time setup cost; subsequent launches reuse it
- **Stream Capture** — `cudaStreamBeginCapture` / `cudaStreamEndCapture` records stream operations into a graph automatically; the runtime infers the same node structure as the manually built graph
- **Graph Clone** — `cudaGraphClone` deep-copies the graph structure; each clone can be independently instantiated and launched, useful when multiple CPU threads need to launch the same graph concurrently
- **Graph reuse** — an instantiated graph is a template you launch repeatedly; because this graph begins with an H2D copy from a host buffer, refilling that buffer before each launch feeds new data through the same graph (with a `cudaStreamSynchronize` between launches so the copy reads the data before the host overwrites it)
- **`cub::BlockReduce`** — CUB's block-scoped reduction primitive; all threads in a block contribute their partial sum and thread 0 receives the block total

## Key APIs

### CUDA Runtime — Explicit Graph Construction
- `cudaGraphCreate` — create an empty graph
- `cudaGraphAddNode` — add a node of any type (memcpy, kernel, host, …) via a `cudaGraphNodeParams` struct
- `cudaGraphNodeParams` — unified node descriptor: a `.type` tag plus a union of per-type parameters
- `cudaGraphGetNodes` — query the number of nodes in a graph

### CUDA Runtime — Stream Capture
- `cudaStreamBeginCapture` — put a stream into capture mode; subsequent operations are recorded, not executed
- `cudaStreamEndCapture` — stop recording and return the captured graph
- `cudaLaunchHostFunc` — schedule a CPU callback on a stream (captured as a host node)

### CUDA Runtime — Instantiation and Launch
- `cudaGraphInstantiate` — compile a graph into an executable `cudaGraphExec_t`
- `cudaGraphLaunch` — submit the entire graph to a stream in a single call
- `cudaGraphClone` — deep-copy a graph structure
- `cudaStreamSynchronize` — block until the stream's queued work (including the graph's H2D copy) completes, so the host input buffer can be safely refilled for the next launch
- `cudaGraphExecDestroy` — release an executable graph
- `cudaGraphDestroy` — release a graph

### CUB
- `cub::BlockReduce<T, BLOCK_THREADS>::Sum` — reduce all per-thread values to a single block sum; result lands on thread 0

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.5 or higher

### Software
- CMake 3.20 or newer
- A C++17-capable host compiler

## How to Build

See the [top-level README](../../../README.md#building-cuda-samples) for full build instructions, including how to build all samples or a single sample standalone.

## How to Run

**Explicit graph construction:**
```bash
./simpleCudaGraphs_explicit
```

**Stream capture:**
```bash
./simpleCudaGraphs_capture
```

## Expected Output

Both executables build the same 5-node graph (`Graph node count: 5`), confirming that stream capture produces the same structure as the explicit construction. Because the host input is refilled with new random values before every launch, **each of the six launches prints a different reduced sum** — showing that one instantiated graph reprocesses fresh data each iteration. The exact values depend on the platform's `rand()` implementation, but they are reproducible run-to-run and both executables print the same sequence.

**`simpleCudaGraphs_explicit`**
```text
GPU Device 0: compute capability X.Y, <smCount> SMs

Reducing 16777216 elements
Threads per block   : 512
Graph launch iterations: 3

=== Explicit Graph Construction ===
Graph node count: 5
[cudaGraphsManual] Host callback final reduced sum = 0.996214
[cudaGraphsManual] Host callback final reduced sum = 0.996187
[cudaGraphsManual] Host callback final reduced sum = 0.996120

Cloned graph:
[cudaGraphsManual] Host callback final reduced sum = 0.996150
[cudaGraphsManual] Host callback final reduced sum = 0.996184
[cudaGraphsManual] Host callback final reduced sum = 0.996056
```

**`simpleCudaGraphs_capture`** prints the same six values, each line labeled `[cudaGraphsUsingStreamCapture]` and under a `=== Stream Capture ===` header.

## Files

- `simpleCudaGraphs.cuh` — shared code: `THREADS_PER_BLOCK`, the `callBackData_t` type, the `myHostNodeCallback` host callback, the `init_input` host helper, and the two reduction kernels (`reduce`, `reduceFinal`)
- `simpleCudaGraphs_explicit.cu` — explicit graph construction via the unified `cudaGraphAddNode` API
- `simpleCudaGraphs_capture.cu` — graph construction via stream capture
- `CMakeLists.txt` — build configuration
- `README.md` — this file

## See Also

- [CUDA Programming Guide — CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs)
- [Technical Blog — Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)
- [CUDA Core Compute Libraries — What is CUB?](https://nvidia.github.io/cccl/unstable/cub/index.html#what-is-cub)
