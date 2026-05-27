# tileLayerNorm

## Description

This sample demonstrates a persistent layer-norm forward pass using
CUDA Tile C++:
`y = (x - mean) * rsqrt(var + eps) * weight + bias`. The grid launches `NUM_SMS`
persistent blocks; each block walks the row dimension with a grid-stride loop,
processing `BLOCK_N` rows by `BLOCK_D` cols per iteration and
striding by `NUM_SMS * BLOCK_N` rows between iterations. Per-row
mean and inverse standard deviation are reduced across the column
dimension with `cuda::tiles` row reductions and saved to float32
side buffers, while the weight and bias tiles are loaded once and
broadcast across rows.

## Expected Output

```
Success! Persistent LayerNorm matches expected results.
```

## Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 13.3 or later.
- [CUDA Driver](https://www.nvidia.com/en-us/drivers/) version 580 or later.
- Host compiler with C++20 support.
