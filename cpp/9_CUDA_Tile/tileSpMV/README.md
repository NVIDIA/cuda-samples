# tileSpMV

## Description

This sample demonstrates sparse matrix-vector multiplication (SpMV)
`y = A * x` using CUDA Tile C++.

The matrix is built directly on the host in Sliced ELLPACK (SELL)
format — the format the Tile kernel actually reads. SELL is the
same idea as ELLPACK applied per-slice: rows are grouped into
slices of `ROWS` consecutive rows (sorted by length to minimize
padding within a slice) and stored column-major so that *the k-th
nonzero of every row in the slice* occupies a contiguous span of
`ROWS` elements in memory.

Each CTA processes one slice using a 2D tile of `shape<ROWS, COLS>`:

- Dimension 0 (`ROWS`): the rows of the slice (one tile row per
  matrix row in the slice)
- Dimension 1 (`COLS`): the next `COLS` nonzeros of every row in the
  slice, processed simultaneously

The kernel computes partial products against the x-vector (an
irreducible gather), accumulates into a 2D tile, reduces along the
column dimension with `cuda::tiles::sum(acc, 1_ic)` to produce one
sum per row, and scatters the per-row sums to `y` using the slice
permutation array.

The sample generates a single random sparse matrix and verifies the
Tile kernel's output against a CPU reference SpMV.

## Expected Output

```
Random sparse matrix: rows=100000, cols=100000, nnz=..., avg nnz/row=...
Tile configuration: ROWS=64, COLS=16 (... slices)
Success! Tile SpMV matches the CPU reference.
```

## Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 13.3 or later.
- [CUDA Driver](https://www.nvidia.com/en-us/drivers/) version 580 or later.
- Host compiler with C++20 support.
