# tileRope

## Description

This sample demonstrates a Rotary Position Embedding (RoPE) forward pass
using CUDA Tile C++. RoPE injects positional information into the query
and key projections of an attention layer by rotating pairs of features
in the head dimension by per-position angles. This implementation uses
the split-half convention: for each token at position `s` the pair
`(q[i], q[i + D/2])` is rotated by `theta = s * 10000^(-2i / D)`, so
`q[i]' = q[i]*cos(theta) - q[i+D/2]*sin(theta)` and
`q[i+D/2]' = q[i]*sin(theta) + q[i+D/2]*cos(theta)`. The
`cuda::tiles::partition_view` type is used to partition each (batch,
position) token's Q and K tensors into 2D tiles over (heads,
half_rope_dim), and a single block processes all heads for one token
in parallel, writing the result back in place. A SIMT kernel is used
to initialize the inputs and the cos/sin tables.

## Expected Output

```
Success! RoPE matches expected results.
```

## Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 13.3 or later.
- [CUDA Driver](https://www.nvidia.com/en-us/drivers/) version 580 or later.
- Host compiler with C++20 support.
