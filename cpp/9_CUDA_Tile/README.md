# 9. CUDA Tile

### [helloTile](./helloTile)

A CUDA Tile C++ sample demonstrating basic usage of tile kernels. This sample shows how to launch a tile kernel and how data can be passed between SIMT and Tile kernels through global device memory.

### [tileVectorAdd](./tileVectorAdd)

This sample demonstrates a simple vector addition using CUDA Tile C++.
The vector addition is performed by splitting the dataset into blocks
which process 1024 elements at a time. The cuda::tiles::partition_view
type is used to partition the data into chunks of size 1024. Each
block loads its respective chunk from 'a' and 'b', performs an
elementwise addition, then stores it to the corresponding chunk of
'c'. Masked loads and stores are used to ensure that the last chunk
which is partially out of bounds is correctly handled.

### [tileTranspose](./tileTranspose)

This sample demonstrates how to transpose a 2D matrix using CUDA Tile
C++. Each block handles an n x m sized chunk of the source matrix. The
block loads a chunk, transposes it locally, and stores it to the
correct position in the result matrix. A cuda::tiles::partition_view
is used to model the chunking of the source and result matrices.

### [tileMatmul](./tileMatmul)

This sample demonstrates how to write a matrix multiplication kernel with good performance in CUDA Tile C++. The kernel multiplies FP16 input tiles with FP32 accumulation using cuda::tiles::mma. The sample compares a naive implementation with an optimized implementation that applies good practices and provides the compiler with additional guidance for better code generation. The host code validates both results and uses CUDA events to compare execution time.

### [tileMatmulAutotuner](./tileMatmulAutotuner)

A CUDA Tile C++ sample demonstrating an nvrtc/nvcc autotuner over a matrix multiplication kernel. This sample shows how autotuning can help guide the choice of tile sizes and optimization hints.

### [tileBmm](./tileBmm)

This sample demonstrates a static-persistent batched matrix multiplication
(BMM) using CUDA Tile C++. Given inputs A of shape (Q, M, K) and B of
shape (Q, K, N), the kernel computes C = A x B of shape (Q, M, N). The
grid launches a fixed number of persistent blocks sized from the device's
SM count, and each block walks the (M, N, Q-chunk) tile space via a
grid-stride loop. Each iteration consumes a chunk of BLOCK_SIZE_Q batches
and issues a single rank-3 batched cuda::tiles::mma per K-step, with the
(M, N) output partitioned into BLOCK_SIZE_M x BLOCK_SIZE_N tiles via
cuda::tiles::partition_view.

### [tileLayerNorm](./tileLayerNorm)

This sample demonstrates a persistent layer-norm forward pass using
CUDA Tile C++: `y = (x - mean) * rsqrt(var + eps) * weight + bias`.
The grid launches `NUM_SMS` persistent blocks; each block walks the
row dimension with a grid-stride loop, processing `BLOCK_N` rows by
`BLOCK_D` cols per iteration. Per-row mean and inverse standard
deviation are reduced across the column dimension with `cuda::tiles`
row reductions, while the weight and bias tiles are loaded once and
broadcast across rows. Compile-time template parameters for `N`,
`D`, `NUM_SMS`, and `EPS` let the tile compiler fold the loop step,
the `(1/D)` reciprocal, partition_view extents, and the eps
broadcast.

### [tileRope](./tileRope)

This sample demonstrates a Rotary Position Embedding (RoPE) forward
pass using CUDA Tile C++. The implementation uses the split-half
(GPT-NeoX style) convention: for each token at position `s` the pair
`(q[i], q[i + D/2])` is rotated by `theta = s * 10000^(-2i / D)`. The
`cuda::tiles::partition_view` type partitions the Q and K tensors
over (heads, half_rope_dim), and a single block processes all heads
for one (batch, position) token in parallel, writing the result back
in place against precomputed cos/sin tables.

### [tileSpMV](./tileSpMV)

This sample demonstrates sparse matrix-vector multiplication (SpMV)
`y = A * x` using CUDA Tile C++.
