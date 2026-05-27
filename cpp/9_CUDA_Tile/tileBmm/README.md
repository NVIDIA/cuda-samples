# tileBmm

## Description

This sample demonstrates a static-persistent batched matrix multiplication
(BMM) using CUDA Tile C++. Given inputs A of shape (Q, M, K) and B of shape
(Q, K, N), the kernel computes C = A x B of shape (Q, M, N). The grid
launches a fixed number of persistent blocks sized from the device's SM
count, and each block walks the (M, N, Q-chunk) tile space via a grid-stride
loop. The batch dimension is tiled by BLOCK_SIZE_Q so every iteration issues
a single rank-3 batched cuda::tiles::mma per K-step over tiles of shape
(BLOCK_SIZE_Q, BLOCK_SIZE_M, BLOCK_SIZE_K) and
(BLOCK_SIZE_Q, BLOCK_SIZE_K, BLOCK_SIZE_N). Grouped ordering on
(pid_m, pid_n) gives L2 reuse. The accumulator is kept in float32 for
precision, and masked loads/stores handle tiles that overhang the matrix
or batch boundaries. Inputs and outputs use __half precision.

## Expected Output

```
Success! BMM matches expected results.
```

## Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 13.3 or later.
- [CUDA Driver](https://www.nvidia.com/en-us/drivers/) version 580 or later.
- Host compiler with C++20 support.
