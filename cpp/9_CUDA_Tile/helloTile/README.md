# helloTile

## Description

This CUDA Tile C++ sample demonstrates basic usage of tile
kernels. This code launches a tile kernel using the triple chevron
syntax and passes data between SIMT and Tile code through global
device memory.

Error checks are performed using `cudaGetLastError` to catch kernel launch issues and `cudaDeviceSynchronize` to catch kernel execution issues.

## Expected Output

```
Hello, SIMT!
[SIMT] *x == 0
[SIMT] *x = 100

Hello, Tile!
[Tile] *x == 100
[Tile] *x = 200

Hello, Host!
[Host] *x == 200
```

## Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 13.3 or later.
- [CUDA Driver](https://www.nvidia.com/en-us/drivers/) version 580 or later.
- Host compiler with C++20 support.
