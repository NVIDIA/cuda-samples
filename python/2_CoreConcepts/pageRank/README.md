# Sample: PageRank Algorithm (Python)

## Description

Demonstrates GPU-accelerated PageRank computation for graph analysis using RAPIDS cuGraph, with cuda.core for device, stream, and GPU timing. This sample focuses on cuda.core integration with high-level libraries (cuGraph/cuDF); for custom kernel programming (Program, LaunchConfig, launch), see the blockwiseSum sample.

## What You'll Learn

- Graph representation using cuDF DataFrames for edge lists
- GPU-optimized PageRank via RAPIDS cuGraph library
- Performance comparison between cuGraph GPU and CPU reference implementation
- cuda.core device/stream management and GPU timing

## Key Libraries

- `cugraph` - RAPIDS GPU-accelerated graph analytics
- `cudf` - RAPIDS GPU DataFrame library
- `cuda.core` - Device, stream, and event APIs for GPU timing
- `cupy` - GPU array library (ExternalStream for cuDF/cuGraph)
- `numpy` - CPU reference implementation

## Key APIs

### From cuda.core:

- `Device(0)` - Create device, `device.set_current()`, `device.create_stream()`
- `EventOptions(enable_timing=True)` - GPU timing via `stream.record()`
- `cp.cuda.ExternalStream(stream.handle).use()` - Make cuDF/cuGraph use cuda.core stream

### From cuGraph:

- `cugraph.Graph(directed=True)` - Create directed graph structure
- `Graph.from_cudf_edgelist()` - Build graph from edge list DataFrame
- `cugraph.pagerank()` - GPU-accelerated PageRank algorithm

### From cuDF:

- `cudf.DataFrame()` - GPU DataFrame for edge lists

## Requirements

### Hardware:

- NVIDIA GPU with Compute Capability 7.0 or higher
- Minimum GPU memory: 512 MB (for 10K node graph)

### Software:

- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- See requirements.txt for package dependencies

## Installation

```bash
cd /path/to/cuda-samples/python/2_CoreConcepts/pageRank
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to Run

```bash
python pageRank.py
```

## Algorithm

The PageRank formula iteratively computes node importance:

```
PR(v) = (1-d)/N + d * Σ PR(u)/out_degree(u)
```

Where:
- `d` = damping factor (typically 0.85)
- `N` = total number of nodes
- Sum is over all nodes `u` that link to `v`

## Expected Output

```
============================================================
PageRank Algorithm (using RAPIDS cuGraph)
============================================================

Device: NVIDIA GeForce RTX ...
Compute Capability: sm_XX

Graph Parameters:
  Nodes: 10,000
  Avg edges/node: 15
  Total edges: ~150,000
  Avg in-degree: 14.9

------------------------------------------------------------
GPU PageRank (RAPIDS cuGraph)
------------------------------------------------------------
Time: X.XXX ms

Top 5 nodes by PageRank:
  1. Node XXXXX: 0.XXXXXX
  ...

------------------------------------------------------------
CPU PageRank (Reference)
------------------------------------------------------------
Time: XXXX.XXX ms
Iterations: XX

------------------------------------------------------------
PERFORMANCE SUMMARY
------------------------------------------------------------
GPU (cuGraph): X.XXX ms
CPU (Reference): XXXX.XXX ms
Speedup: XXXX.Xx

------------------------------------------------------------
VERIFICATION
------------------------------------------------------------
GPU vs CPU PageRank scores: Test PASSED

PageRank Properties:
  Sum of scores: 1.000000 (should be ~1.0)
  Sum check: ✓

Done
```

## Files

- `pageRank.py` - Python implementation using RAPIDS cuGraph
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## Why cuGraph?

RAPIDS cuGraph provides production-grade, GPU-accelerated graph analytics:

- **Highly optimized** - Uses advanced GPU parallelization techniques
- **Scalable** - Handles graphs with billions of edges
- **Easy to use** - Simple Python API similar to NetworkX
- **Integrated** - Works seamlessly with cuDF, cuML, and other RAPIDS libraries

## Applications

- Web page ranking (original Google PageRank)
- Social network influence analysis
- Citation network analysis
- Recommendation systems
- Fraud detection in financial networks

## See Also

- [RAPIDS cuGraph Documentation](https://docs.rapids.ai/api/cugraph/stable/)
- [cuGraph GitHub Repository](https://github.com/rapidsai/cugraph)
- [RAPIDS Installation Guide](https://rapids.ai/start.html)
