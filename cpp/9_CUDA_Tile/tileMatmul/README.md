# tileMatmul

## Description

This sample demonstrates how to write a matrix multiplication kernel with good
performance in CUDA Tile C++. The kernel multiplies FP16 input tiles with FP32
accumulation using `cuda::tiles::mma`.

The sample compares a naive implementation with an optimized implementation
that applies good practices and provides the compiler with additional guidance
for better code generation. The host code uses CUDA events to capture execution time.

The optimized kernel uses neutral placeholder values for `LOAD_LATENCY` and
`STORE_LATENCY`. See [tileMatmulAutotuner](../tileMatmulAutotuner) for an
example of autotuning these values to find a suitable
configuration for your hardware.

## Running

```
# Run with default warmup and benchmark iterations. Validation is disabled by default.
./tileMatmul

# Enable CPU validation
./tileMatmul --validate

# To run faster, skip warmups (not recommended) and set iteration to 1
./tileMatmul --skip-warmup -i 1

# Show all options
./tileMatmul --help
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--validate` | Enable CPU cross-validation. Validation is disabled by default. |
| `--skip-warmup` | Disable warmup iterations. |
| `--warmup=N` | Set warmup iterations. The default is 5. |
| `-i N`, `--iters=N` | Set benchmark iterations. The default is 20. |

## Example Output

```
Note: CPU cross-validation disabled
  matmul                                    :   0.073 ms,   115.2 GB/s, 29495.8 GFLOPS 
  matmul_naive                              :   0.497 ms,    16.9 GB/s, 4322.8 GFLOPS
```

## Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 13.3 or later.
- [CUDA Driver](https://www.nvidia.com/en-us/drivers/) version 580 or later.
- Host compiler with C++20 support.
