# tileVectorAdd

## Description

This sample demonstrates a simple vector addition using CUDA Tile C++.
The vector addition is performed by splitting the dataset into blocks
which process 1024 elements at a time. The cuda::tiles::partition_view
type is used to partition the data into chunks of size 1024. Each
block loads its respective chunk from 'a' and 'b', performs an
elementwise addition, then stores it to the corresponding chunk of
'c'. Masked loads and stores are used to ensure that the last chunk
which is partially out of bounds is correctly handled.

## Expected Output

```
Success! Vector addition matches expected results.
```

## Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 13.3 or later.
- [CUDA Driver](https://www.nvidia.com/en-us/drivers/) version 580 or later.
- Host compiler with C++20 support.
