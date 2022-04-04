/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


// CUDA sample demonstrating a Double precision GEMM computation using the Warp
//  Matrix Multiply and Accumulate API introduced in CUDA 11.0.

// In this program, the compute_dgemm kernel computes the result of a matrix multiplication
// and addition: D = alpha * A * B + beta * C. The dimensions of both C and D matrices
// are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x K_GLOBAL (row-major), the B matrix
// is K_GLOBAL x N_GLOBAL (column-major).
// In that kernel, each CTA computes one 64 x 64 tile of the resulting matrix
// per iteration. When the tile is computed, the CTA stores it to the global memory
// and begins a new iteration, selecting a new 64 x 64 tile to compute.
// Each CTA consists of eight warps. For the 64 x 64 tile, each warp computes eight
// 8 x 8 subtiles, organized in a 2 x 4 two-dimensional array.
// Warps compute the 8 x 8 subtiles using nvcuda::wmma::mma_sync operations by
// moving through the K_GLOBAL dimension of the A and B matrices and accumulating
// the intermediate result in the local thread state.

// There are a number of simple optimizations used in the algorithm:
// - The CTA copies the 64 x 64 tile of the C matrix from the global memory to
//   shared memory. After that is done, each warp loads the C matrix fragments from
//   shared memory, thus avoiding a random global memory access.
// - On each internal iteration, the CTA copies a portion of the A and B matrices from
//   global memory to shared memory. After that, all warps in the CTA reuse the A and B
//   data from shared memory, thus reducing the number of data copies from global memory.
// - The portions of the A and B matrices are stored in shared memory with an additional
//   padding (skew) to reduce the number of shared memory access bank conflicts.
//   (See a detailed explanation near the SKEW_DOUBLE macro definition.)
// - When the CTA finishes computing the tiles of the resulting matrix, each warp stores
//   its subtiles to shared memory. The CTA then copies the shared memory contents to
//   global memory, again avoiding redundant random global memory accesses.
// - Note that the CTA tile size is chosen to maximize the GPU register utilization,
//   but carefully enough to avoid local memory use.

#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/std/type_traits>
#include <cuda/barrier>
#include <cuda/pipeline>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

// Externally configurable parameters.

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 0
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 8
#define N 8
#define K 4

// GEMM configuration.

#define M_TILES 1024
#define N_TILES 1024
#define K_TILES 1024

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit 8x16-tile chunks of each
// the A and B matrix data, that are (M = 8) * (K = 4) * 8 * (CHUNK_K = 16) * sizeof(double) = 32 Kb each
// But we cannot account the 4 Kb total skew overhead, without which the performance
// would be severely impacted. So we choose to reduce the chunk size in half,
// i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 8
#else
#define CHUNK_K 16
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(double))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B matrix
// in shared memory to minimize possible bank conflicts.
// Before performing the nvcuda::wmma::mma_sync operation, the warp must load the matrix
// data using the nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
// is not specified for that function, each lane in the warp can read one or multiple matrix
// elements from different matrix rows or columns.
// For shared memory, such access can result in bank conflicts if different rows / columns
// of the matrix map to the same bank. By shifting each row and column by a few bytes, we
// make sure that they map to different banks, thus reducing the number of possible bank
// conflicts.
// The number of 4 eight-byte "double" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
#define SKEW_DOUBLE 4

#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)

enum kernels
{
    dmma_shmem_gemm_async_copy      = 0, // DMMA shmem using kernel with async_copy
    dmma_shmem_gemm_cg_async_copy   = 1, // DMMA shmem using kernel with cooperative groups async_copy
    dmma_shmem_gemm                 = 2, // DMMA shmem using kernel normal copy (without async_copy).
    simple_dmma_gemm                = 3  // DMMA non-shmem using simple kernel.
};

const char* kernelNames[] = {"compute_dgemm_async_copy", "compute_dgemm_cg_async_copy",
                            "compute_dgemm", "simple_wmma_gemm"};

using namespace nvcuda;
namespace cg = cooperative_groups;

__host__ void init_host_matrices(double *a, double *b, double *c)
{
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            a[i*K_GLOBAL+j] = (double) (rand() % 3);
        }
    }

    for (int i = 0; i < N_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            b[i*K_GLOBAL+j] = (double) (rand() % 3);
        }
    }

    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        c[t] =  (double) (rand() % 3);
    }
}

__global__ void compute_dgemm(const double *A, const double *B, const double *C, double *D, double alpha, double beta)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ double shmem[][CHUNK_K * K + SKEW_DOUBLE];

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Offset in shared memory from which the B matrix is stored.
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;


    // This pointer is used to access the C and D matrix tiles this warp computes.
    double *shmem_warp_tile_ptr = (double*)&shmem[0][0] + (warpId / BLOCK_ROW_WARPS) * SHMEM_STRIDE * N * BLOCK_ROW_WARPS + (warpId % BLOCK_ROW_WARPS) * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
    double *shmem_warp_stream_ptr = (double*)&shmem[0][0] + warpId * SHMEM_STRIDE * N;

    // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    // each tile computation. Technically this is not generally correct (may result
    // in a loss of precision). Zero still needs to be specially handled though.
    beta /= alpha;

    // Each CTA slides along the 64 x 64 tiles from the top left corner of the matrix to the
    // right and down, and selects the next tile to compute. Once there's no such tile,
    // all warps in this CTA exit.
    for(unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES) {
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from to shared memory.
        const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        const double *src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory.
#pragma unroll
        for (int i = 0; i < N; i++) {
            *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                *((int4 *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId);
        }

        __syncthreads();

        // These fragments will accumulate the result of A and B matrix fragment multiplications
        // along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, M, N, K, double> c[WARP_COL_TILES][WARP_ROW_TILES];

        // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                const double *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * N + j * N;

                wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Scale the C matrix.
#pragma unroll
       for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++) {
                    c[i][j].x[t] *= beta;
                }
            }
        }

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const double *warp_ptr = (warpId < (WARPS_PER_BLOCK/2)) ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK/2)) * 2) :
                                              (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK/2)) * 2);

        // Go through the global K dimension by a fixed step at a time.
#pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, the rest copy the B matrix.
            size_t shmem_idx = warpId < (WARPS_PER_BLOCK/2) ? (M * (warpId % (WARPS_PER_BLOCK/2)) * 2) :
                                                              (N * (warpId % (WARPS_PER_BLOCK/2)) * 2 + shmem_idx_b_off);

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            const double *lane_ptr = warp_ptr + tile_k * K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL;

            // Shift the second half of the warp to the next row / column in the shared memory.
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
            for(int i = 0; i < ((WARP_SIZE/2) / CHUNK_COPY_LINES_PER_WARP); i++) {
                 // Copy 16 bytes at once in each lane.
                *((int4*)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) = *((int4*)lane_ptr +  (laneId % CHUNK_COPY_LINE_LANES));

                // Advance the global memory pointer and the shared memory index.
                lane_ptr = lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP;
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }

            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
#pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, double, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t shmem_idx_a = (warpId/2) * M * 2 + (i * M);
                    const double *tile_ptr = &shmem[shmem_idx_a][k_step * K];

                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_DOUBLE);

#pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be reused
                            // against the other A matrix fragments.
                            size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId%2) + (j * N);
                            const double *tile_ptr = &shmem[shmem_idx_b][k_step * K];

                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_DOUBLE);

                        }

                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }

            __syncthreads();
        }

        // Store the D fragments to shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                // Uniform, point-wise transformations of ALL fragment elements by ALL threads in the
                // warp are well-defined even though element indices within fragment storage are not defined.
#pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++)
                    c[i][j].x[t] *= alpha;

                double *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * N + j * N;

                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Now that shared memory contains all the D tiles, stream them to global memory.
        double *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
        for (int i = 0; i < N; i++) {
            *((int4*)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                *((int4*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }

        __syncthreads();
    }
#endif
}

__global__ void compute_dgemm_async_copy(const double *A, const double *B, const double *C, double *D, double alpha, double beta)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ double shmem[][CHUNK_K * K + SKEW_DOUBLE];

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Offset in shared memory from which the B matrix is stored.
    constexpr size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    // This pointer is used to access the C and D matrix tiles this warp computes.
    double *shmem_warp_tile_ptr = &shmem[0][0] + (warpId/BLOCK_ROW_WARPS) * SHMEM_STRIDE * N * BLOCK_ROW_WARPS + (warpId % BLOCK_ROW_WARPS) * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
    double *shmem_warp_stream_ptr = &shmem[0][0] + warpId * SHMEM_STRIDE * N;

    // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    // each tile computation. Technically this is not generally correct (may result
    // in a loss of precision). Zero still needs to be specially handled though.
    beta /= alpha;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    const auto shape2 = cuda::aligned_size_t<alignof(double2)>(sizeof(double2));
    constexpr int loadStride = 1; // load 2 double, left-shift by 1.

    // Each CTA slides along the 64 x 64 tiles from the top left corner of the matrix to the
    // right and down, and selects the next tile to compute. Once there's no such tile,
    // all warps in this CTA exit.
    for(unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES) {
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from to shared memory.
        const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        const double *src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory.
#pragma unroll
        for (int i = 0; i < N; i++) {
            pipe.producer_acquire();
            cuda::memcpy_async(&shmem_warp_stream_ptr[(SHMEM_STRIDE * i) + (laneId << loadStride)],
                                &src_gmem_warp_stream_ptr[(GLOBAL_MEM_STRIDE * i) + (laneId << loadStride)],
                                shape2, pipe);

            pipe.producer_commit();
        }
        // Now wait for all the above issued 8 batches to complete.
        cuda::pipeline_consumer_wait_prior<0>(pipe);
        __syncthreads();

        // These fragments will accumulate the result of A and B matrix fragment multiplications
        // along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, M, N, K, double> c[WARP_COL_TILES][WARP_ROW_TILES];

        // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                const double *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * N + j * N;

                wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
                // Scale the C matrix.
#pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++) {
                    c[i][j].x[t] *= beta;
                }
            }
        }

        pipe.consumer_release();
        // sync here so that shared memory can then be used for loading A & B matrices.
        __syncthreads();

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const double *warp_ptr = (warpId < (WARPS_PER_BLOCK/2)) ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK/2)) * 2) :
                                              (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK/2)) * 2);

        const int stridePerLaneCopy = (laneId / CHUNK_COPY_LINE_LANES);
        constexpr int chunksPerLane = ((WARP_SIZE/2) / CHUNK_COPY_LINES_PER_WARP);
        const int laneLoadElem = (laneId % CHUNK_COPY_LINE_LANES) << loadStride;

        // Go through the global K dimension by a fixed step at a time.
#pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, the rest copy the B matrix.
            // As for DMMA  M == N we use M for warp 4-7 + shmem_idx_b_off.
            size_t shmem_idx = (M * (warpId % (WARPS_PER_BLOCK/2)) * 2) + (shmem_idx_b_off * (warpId/(WARPS_PER_BLOCK/2)));

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            const double *lane_ptr = warp_ptr + tile_k * K + stridePerLaneCopy * K_GLOBAL + laneLoadElem;

            // Shift the second half of the warp to the next row / column in the shared memory.
            shmem_idx += stridePerLaneCopy;
#pragma unroll
            for(int i = 0; i < chunksPerLane; i++) {
                 // Copy 16 bytes at once in each lane.
                pipe.producer_acquire();

                cuda::memcpy_async(&shmem[shmem_idx][laneLoadElem], lane_ptr, shape2, pipe);

                pipe.producer_commit();

                // Advance the global memory pointer and the shared memory index.
                lane_ptr = lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP;
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }

            cuda::pipeline_consumer_wait_prior<0>(pipe);
            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
#pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, double, wmma::col_major> b[WARP_ROW_TILES];
#pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t shmem_idx_a = (warpId/2) * M * 2 + (i * M);
                    const double *tile_ptr = &shmem[shmem_idx_a][k_step * K];

                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_DOUBLE);
#pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be reused
                            // against the other A matrix fragments.
                            size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId%2) + (j * N);
                            const double *tile_ptr = &shmem[shmem_idx_b][k_step * K];

                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_DOUBLE);
                        }
                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }
            pipe.consumer_release();
            __syncthreads();
        }

        // Store the D fragments to shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                // Uniform, point-wise transformations of ALL fragment elements by ALL threads in the
                // warp are well-defined even though element indices within fragment storage are not defined.
#pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++)
                    c[i][j].x[t] *= alpha;

                double *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * N + j * N;

                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Now that shared memory contains all the D tiles, stream them to global memory.
        double *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
        for (int i = 0; i < N; i++) {
            *((int4*)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                *((int4*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }

        __syncthreads();
    }
#endif
}

 __global__ void compute_dgemm_cg_async_copy(const double *A, const double *B, const double *C, double *D, double alpha, double beta)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ double shmem[][CHUNK_K * K + SKEW_DOUBLE];
    auto cta = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(cta);

    constexpr int tileChunkCopySize = WARP_SIZE / CHUNK_COPY_LINES_PER_WARP;
    auto tileChunkCopy = cg::tiled_partition<tileChunkCopySize>(cta);

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Offset in shared memory from which the B matrix is stored.
    constexpr size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    // This pointer is used to access the C and D matrix tiles this warp computes.
    double *shmem_warp_tile_ptr = (double*)&shmem[0][0] + (warpId/2) * SHMEM_STRIDE * N * 2 + (warpId%2) * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
    double *shmem_warp_stream_ptr = (double*)&shmem[0][0] + warpId * SHMEM_STRIDE * N;

    // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    // each tile computation. Technically this is not generally correct (may result
    // in a loss of precision). Zero still needs to be specially handled though.
    beta /= alpha;

    // Each CTA slides along the 64 x 64 tiles from the top left corner of the matrix to the
    // right and down, and selects the next tile to compute. Once there's no such tile,
    // all warps in this CTA exit.
    for(unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES) {
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from to shared memory.
        const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        const double *src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory.
#pragma unroll
        for (int i = 0; i < N; i++) {
            auto dst_ptr = &shmem_warp_stream_ptr[(SHMEM_STRIDE * i)];
            auto src_ptr = &src_gmem_warp_stream_ptr[(GLOBAL_MEM_STRIDE * i)];
            cg::memcpy_async(tile32, dst_ptr, src_ptr, cuda::aligned_size_t<alignof(double2)>{tile32.size() * sizeof(double2)});
        }

        cg::wait(cta);

        // These fragments will accumulate the result of A and B matrix fragment multiplications
        // along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, M, N, K, double> c[WARP_COL_TILES][WARP_ROW_TILES];

        // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                const double *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * N + j * N;
                wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
            }
        }

        // Scale the C matrix.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++) {
                    c[i][j].x[t] *= beta;
                }
            }
        }

        // sync here so that shared memory can then be used for loading A & B matrices.
        cg::wait(cta);
        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const double *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK/2)) * 2) :
            (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK/2)) * 2);

        const int stridePerLaneCopy = (laneId / CHUNK_COPY_LINE_LANES);
        constexpr int chunksPerLane = ((WARP_SIZE/2) / CHUNK_COPY_LINES_PER_WARP);
        // Go through the global K dimension by a fixed step at a time.
#pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, the rest copy the B matrix.
            // As for DMMA  M == N we use M for warp 4-7 + shmem_idx_b_off.
            size_t shmem_idx = (M * (warpId % (WARPS_PER_BLOCK/2)) * 2) + (shmem_idx_b_off * (warpId/(WARPS_PER_BLOCK/2)));

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            auto lane_ptr = warp_ptr + tile_k * K + stridePerLaneCopy * K_GLOBAL;

            // Shift the second half of the warp to the next row / column in the shared memory.
            shmem_idx += stridePerLaneCopy;

#pragma unroll
            for(int i = 0; i < chunksPerLane; i++) {
                // Copy 16 bytes at once in each lane.
                auto dst_ptr = &shmem[shmem_idx][0];
                auto src_ptr = lane_ptr;

                cg::memcpy_async(tileChunkCopy, dst_ptr, src_ptr, 
                                cuda::aligned_size_t<alignof(double2)>{tileChunkCopySize * sizeof(double2)});

                // Advance the global memory pointer and the shared memory index.
                lane_ptr = lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP;
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }
            cg::wait(cta);

            // Compute a grid of C matrix tiles in each warp.
#pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, double, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t shmem_idx_a = (warpId/2) * M * 2 + (i * M);
                    const double *tile_ptr = &shmem[shmem_idx_a][k_step * K];

                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_DOUBLE);

#pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be reused
                            // against the other A matrix fragments.
                            size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId%2) + (j * N);
                            const double *tile_ptr = &shmem[shmem_idx_b][k_step * K];

                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_DOUBLE);

                        }

                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }
            cg::sync(cta);
        }

        // Store the D fragments to shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                // Uniform, point-wise transformations of ALL fragment elements by ALL threads in the
                // warp are well-defined even though element indices within fragment storage are not defined.
#pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++)
                    c[i][j].x[t] *= alpha;

                double *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * N + j * N;

                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }

        cg::sync(cta);

        // Now that shared memory contains all the D tiles, stream them to global memory.
        double *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
        for (int i = 0; i < N; i++) {
            *((int4*)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                *((int4*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }
        cg::sync(cta);
    }
#endif
}

// Performs an MxNxK DGEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 8, 8 and 4 respectively. 
//  3) A is row major, B is column major matrix.
// Note: This is a less performant version of the compute_dgemm kernel. It is designed for
//       demonstration purposes only to show the CUDA WMMA API use without relying on
//       availability of the shared memory.
__global__ void simple_wmma_gemm(double *a, double *b, double *c, double *d, int m_ld, int n_ld, int k_ld, double alpha, double beta)
{
#if __CUDA_ARCH__ >= 800
    // Leading dimensions. Packed with no transpositions.
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, double, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, double> acc_frag;
    wmma::fragment<wmma::accumulator, M, N, K, double> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < k_ld; i += K) {
        int aCol = i;
        int aRow = warpM * M;

        int bCol = warpN * N;
        int bRow = i;

        // Bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int cCol = warpN * N;
    int cRow = warpM * M;

    if (cRow < m_ld && cCol < n_ld) {
        wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
    }
#endif
}

__host__ void matMultiplyOnHost(double *A, double *B, double *C,
                                float alpha, float beta,
                                int numARows, int numAColumns,
                                int numBRows, int numBColumns,
                                int numCRows, int numCColumns)
{
    for (int i = 0; i < numCRows; i++) {
        for (int j = 0; j < numCColumns; j++) {
            double temp = 0.0;

            for (int k = 0; k < numAColumns; k++) {
                // B matrix is column major. A matrix is row major.
                temp += A[i * numAColumns + k] * B[j * numBRows + k];
            }

            C[i*numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
        }
    }
}

int main(int argc, char **argv)
{
    printf("Initializing...\n");

    int dev = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    // Double precision Tensor cores require a GPU of Ampere (SM8X) architecture or higher.
    if (deviceProp.major < 8) {
        printf("dmmaTensorCoreGemm requires SM 8.0 or higher.  Exiting...\n");
        exit(EXIT_WAIVED);
    }

    printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
    printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
    printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

    double *A_h = NULL;
    double *B_h = NULL;
    double *C_h = NULL;
#if CPU_DEBUG
    double *result_hD = NULL;
    double *result_host = NULL;
#endif

    A_h = (double*) malloc(sizeof(double) * M_GLOBAL * K_GLOBAL);
    B_h = (double*) malloc(sizeof(double) * K_GLOBAL * N_GLOBAL);
    C_h = (double*) malloc(sizeof(double) * M_GLOBAL * N_GLOBAL);
#if CPU_DEBUG
    result_hD   = (double*) malloc(sizeof(double) * M_GLOBAL * N_GLOBAL);
    result_host = (double*) malloc(sizeof(double) * M_GLOBAL * N_GLOBAL);
#endif

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    double *D = NULL;

    checkCudaErrors(cudaMalloc((void**)&A, sizeof(double) * M_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc((void**)&B, sizeof(double) * N_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc((void**)&C, sizeof(double) * M_GLOBAL * N_GLOBAL));
    checkCudaErrors(cudaMalloc((void**)&D, sizeof(double) * M_GLOBAL * N_GLOBAL));

    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);

    init_host_matrices(A_h, B_h, C_h);

    printf("Preparing data for GPU...\n");

    checkCudaErrors(cudaMemcpy(A, A_h, sizeof(double) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, sizeof(double) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(C, C_h, sizeof(double) * M_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(D, 0, sizeof(double) * M_GLOBAL * N_GLOBAL));

    enum {
        // Compute the right amount of shared memory to request.
        // We need shared memory to hold per-CTA C and D matrix tiles, and to cache per-CTA chunks
        // of the A and B matrices. Therefore, the right amount to request is the maximum of those
        // two numbers.
        SHMEM_SZ = MAX(sizeof(double) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_DOUBLE) * 2,
                       M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(double))
    };

    printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

    const double alpha = 1.1f;
    const double beta = 1.2f;

    cudaEvent_t start, stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    kernels selected_kernel = dmma_shmem_gemm_async_copy;

    // kernel to run - default (dmma_shmem_gemm_async_copy == 0)
    if (checkCmdLineFlag(argc, (const char **)argv, "kernel")) {
        int kernel_number = getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
        if (kernel_number < 4)
        {
            selected_kernel = (kernels)kernel_number;
        }
        else
        {
            printf("Error: kernel number should be between 0 to 3, you have entered %d\n", kernel_number);
            exit(EXIT_FAILURE);
        }
    }

    // If enough shared memory available on the GPU use high performant kernel
    if ((deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ) && (selected_kernel != simple_dmma_gemm))
    {
        printf("Computing using high performance kernel = %d - %s\n", selected_kernel, kernelNames[selected_kernel]);

        switch (selected_kernel)
        {
            case dmma_shmem_gemm_async_copy :
            default:
                checkCudaErrors(cudaFuncSetAttribute(compute_dgemm_async_copy, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
                checkKernelErrors((compute_dgemm_async_copy<<<deviceProp.multiProcessorCount*3, THREADS_PER_BLOCK, SHMEM_SZ>>>(A, B, C, D, alpha, beta)));
                break;
            case dmma_shmem_gemm_cg_async_copy :
                checkCudaErrors(cudaFuncSetAttribute(compute_dgemm_cg_async_copy, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
                checkKernelErrors((compute_dgemm_cg_async_copy<<<deviceProp.multiProcessorCount*3, THREADS_PER_BLOCK, SHMEM_SZ>>>(A, B, C, D, alpha, beta)));
                break;
            case dmma_shmem_gemm :
                checkCudaErrors(cudaFuncSetAttribute(compute_dgemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
                checkKernelErrors((compute_dgemm<<<deviceProp.multiProcessorCount*2, THREADS_PER_BLOCK, SHMEM_SZ>>>(A, B, C, D, alpha, beta)));
                break;
        }

#if CPU_DEBUG
        checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(double)*M_GLOBAL*N_GLOBAL, cudaMemcpyDeviceToHost));
#endif
    }
    else
    {
        dim3 gridDim;
        dim3 blockDim;

        // blockDim.x must be a multple of warpSize
        // 128x4 means we have 16 warps and a block computes a 64x64 output tile
        blockDim.x = 128;
        blockDim.y = 4;

        gridDim.x = (M_GLOBAL + (M * blockDim.x / 32 - 1)) / (M * blockDim.x / 32);
        gridDim.y = (N_GLOBAL + N * blockDim.y - 1) / (N * blockDim.y);

        printf("Computing... using simple_wmma_gemm kernel\n");
        simple_wmma_gemm<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);
#if CPU_DEBUG
        checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(double) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));
#endif
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

#if CPU_DEBUG
    printf("Verifying correctness of the computations...\n");

    memcpy(result_host, C_h, sizeof(double) * M_GLOBAL * N_GLOBAL);

    matMultiplyOnHost(A_h, B_h, result_host,
                      alpha, beta,
                      M_GLOBAL, K_GLOBAL,
                      K_GLOBAL, N_GLOBAL,
                      M_GLOBAL, N_GLOBAL);

    size_t number_of_matches = 0;
    for (int i = 0; i < N_GLOBAL*M_GLOBAL; i++) {
        if  (fabs(result_hD[i] - result_host[i]) > 0.1f)
        {
            printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i], result_host[i]);
            break;
        }
        else
        {
            number_of_matches++;
        }
    }
    printf("number_of_matches = %zu out of = %d \n", number_of_matches, N_GLOBAL*M_GLOBAL);
    free(result_hD);
    free(result_host);
#endif

    float milliseconds = 0;

    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Time: %f ms\n", milliseconds);
    printf("FP64 TFLOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL * 2)/(milliseconds/1000.)) / 1e12);

    free(A_h);
    free(B_h);
    free(C_h);
    checkCudaErrors(cudaFree((void*)A));
    checkCudaErrors(cudaFree((void*)B));
    checkCudaErrors(cudaFree((void*)C));
    checkCudaErrors(cudaFree((void*)D));

    return 0;
}
