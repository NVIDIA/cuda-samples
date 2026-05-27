# tileTranspose

## Description

This sample demonstrates how to transpose a 2D matrix using CUDA Tile
C++. Each block handles an n x m sized chunk of the source matrix. The
block loads a chunk, transposes it locally, and stores it to the
correct position in the result matrix. A cuda::tiles::partition_view
is used to model the chunking of the source and result matrices.

## Expected Output

```
Success! Matrix transpose matches expected results.
```

## Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 13.3 or later.
- [CUDA Driver](https://www.nvidia.com/en-us/drivers/) version 580 or later.
- Host compiler with C++20 support.
