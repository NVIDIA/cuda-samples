# cudaGraphs (Python)

## Description

This sample demonstrates how to capture a multi-stage kernel pipeline as a
CUDA graph with `cuda.core` and replay it with a single driver call.

The sample runs a three-stage elementwise pipeline
`r3 = (a + b) * c - a` in two modes:

1. **Individual launches** - one `launch(stream, ...)` per stage, repeated
   for every iteration of the pipeline.
2. **CUDA graph replay** - the same three launches are recorded into a
   `Graph` once and replayed with `graph.launch(stream)` on each
   iteration.

Both paths are timed over N iterations and their results are verified
against a reference computation. The sample also re-launches the graph
after mutating the input buffers to show that the graph captures
pointers (not data), so the same graph can process new inputs without
rebuilding.

## What You'll Learn

- Creating a `GraphBuilder` from a stream with `stream.create_graph_builder()`
- Capturing launches with `begin_building()` and `end_building()`
- Completing a graph with `builder.complete()` and uploading it to a stream
- Replaying the graph with `graph.launch(stream)`
- Measuring the launch-overhead savings for small kernels
- Re-running the same graph against updated input data

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - Pythonic access to CUDA runtime, programs, and graphs
- `cupy` - input buffers and result verification
- `numpy` - scalar kernel arguments

## Key APIs

### From `cuda.core`

- `Stream.create_graph_builder()` - obtain a `GraphBuilder`
- `GraphBuilder.begin_building()` / `end_building()` - begin and finish recording launches issued against the builder
- `GraphBuilder.complete()` - produce an executable `Graph`
- `Graph.upload(stream)` - upload the graph structure to the device
- `Graph.launch(stream)` - replay the entire graph
- `launch(graph_builder, config, kernel, ...)` - record a kernel launch into the graph being built

### From `cuda_samples_utils`

- `print_gpu_info()` - print device name and compute capability

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.0 or higher
- Minimum GPU memory: 512 MB

### Software

- CUDA Toolkit 13.0 or newer (matches `cuda-python` 13.x)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=0.6.0)
- `cupy-cuda13x` (>=13.0.0)

## Installation

Install the required packages from `requirements.txt`:

```bash
cd /path/to/cuda-samples/python/2_CoreConcepts/cudaGraphs
pip install -r requirements.txt
```

The `requirements.txt` installs:

- `cuda-python` (>=13.0.0)
- `cuda-core` (>=0.6.0)
- `cupy-cuda13x` (>=13.0.0)

## How to Run

### Basic usage

```bash
cd cuda-samples/python/2_CoreConcepts/cudaGraphs
python cudaGraphs.py
```

### With custom parameters

```bash
# Larger vectors and more iterations
python cudaGraphs.py --elements 4096 --iters 2000

# Use a specific GPU
python cudaGraphs.py --device 1
```

Short vectors exaggerate the launch-overhead savings; larger vectors
will show the two approaches converging because per-launch overhead
becomes negligible next to kernel runtime.

## Expected Output

Speedup numbers vary with GPU and host CPU.

```
Device: <Your GPU Name>
Compute Capability: <X.Y>

Individual launches: 1000 iters in 0.0085s  (8.49 us/iter)

Building CUDA graph...
Graph replay:       1000 iters in 0.0034s  (3.41 us/iter)
Graph speedup: 2.49x

Graph replay on updated data verified (same graph, new buffer contents)

Done
```

**Note:** Device name, compute capability, and speedup will vary based on
your GPU and host CPU.

## Files

- `cudaGraphs.py` - Python implementation using `cuda.core` CUDA graphs
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_samples_utils.py` - Common utilities (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.core` graphs API](https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#cuda-graphs)
- Upstream `cuda.core` example: [`cuda_graphs.py`](https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/cuda_graphs.py)
- [CUDA Graphs programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
