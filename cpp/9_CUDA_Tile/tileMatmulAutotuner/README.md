# tileMatmulAutotuner

## Description

A CUDA Tile C++ sample demonstrating how to autotune tile sizes and optimization hints for matrix multiplication with FP16 inputs and FP32 accumulation when compiling with nvrtc or nvcc.

The sample explores combinations of tile sizes and optimization hints, compiles the Tile C++ matrix multiplication kernel, executes it, and reports the best measured configuration. The launch grid is derived from the selected tile size and matrix dimensions.
The search space is read from `autotuner_search_space.conf` so one can edit tile sizes and hint values without rebuilding the sample.

## Running

```
# Run autotuner. Validation is disabled by default.
./tileMatmulAutotuner

# Select a backend
./tileMatmulAutotuner --backend=nvrtc
./tileMatmulAutotuner --backend=nvcc

# Enable CPU validation
./tileMatmulAutotuner --validate

# To run faster, skip warmups and set iteration to 1
./tileMatmulAutotuner --skip-warmup -i 1

# Show all options
./tileMatmulAutotuner --help
```

NOTE: When using the nvrtc backend, the libnvrtc library must be on the dynamic library search path. This may require adding the CUDA toolkit 'lib' directory to the LD_LIBRARY_PATH (for linux) or PATH (for windows) env var.

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--validate` | Enable CPU cross-validation. Validation is disabled by default. |
| `--skip-warmup` | Disable warmup iterations. |
| `--warmup=N` | Set warmup iterations. The default is 5. |
| `-i N`, `--iters=N` | Set benchmark iterations. The default is 20. |
| `--backend=nvrtc\|nvcc` | Select the backend. The default is `nvrtc`. |

## Search Space Configuration

Each `tile` line adds one `TILE_BLOCK_M`, `TILE_BLOCK_N`, and `TILE_BLOCK_K` combination. The `load_latency` and `store_latency` lines list values that are combined with every `tile` entry.

```
tile 64 64 32
tile 128 64 32
load_latency 2 5 8
store_latency 2 5 8
```

Lines beginning with `#` are comments.

## Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 13.3 or later.
- [CUDA Driver](https://www.nvidia.com/en-us/drivers/) version 580 or later. The NVRTC backend invokes `tileiras` to compile Tile IR to cubin. JIT-compiling Tile IR to cubin with the CUDA Driver API instead requires version 590 or later.
- Host compiler with C++20 support.
