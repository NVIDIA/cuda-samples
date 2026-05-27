/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

/*
 * This sample demonstrates sparse matrix-vector multiplication
 * (SpMV) using CUDA Tile C++.
 *
 *   y = A * x
 *
 * The matrix is built directly on the host in Sliced ELLPACK (SELL)
 * format — the format the Tile kernel actually reads. SELL is the
 * same idea as ELLPACK applied per-slice: rows are grouped into
 * slices of SLICE_ROWS consecutive rows (sorted by length to
 * minimize padding) and stored column-major so that "the k-th
 * nonzero of every row in the slice" occupies contiguous memory.
 *
 *   slice s contains rows row_perm[s*SLICE_ROWS .. s*SLICE_ROWS + SLICE_ROWS)
 *   sell_data[slice_offsets[s] + k * SLICE_ROWS + r] is the k-th
 *   nonzero of the r-th row of slice s, with padding (column = 0,
 *   value = 0.0f) when the row has fewer than slice_widths[s]
 *   nonzeros
 *
 * Each CTA processes one slice using a 2D tile of shape<ROWS, COLS>:
 *
 *   - Dimension 0 (ROWS): the rows of the slice (one tile row per
 *     matrix row in the slice)
 *   - Dimension 1 (COLS): the next COLS nonzeros of every row in
 *     the slice, processed simultaneously
 *
 * Because of the column-major slice layout, "load the next COLS
 * nonzeros for every row" is a contiguous, fully coalesced load.
 * This is the property that distinguishes SELL from a row-by-row
 * CSR kernel: CSR forces the kernel to gather column indices and
 * values from disjoint per-row address streams. The kernel computes
 * partial products against the x-vector (an irreducible gather),
 * accumulates into a 2D tile, then reduces along the column
 * dimension with 'cuda::tiles::sum(acc, 1_ic)' to produce one sum
 * per row, and scatters the sums to y using row_perm.
 */

#include "helper_cuda.h"

#include "cuda_tile.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace ct = cuda::tiles;

//=============================================================================
// Tile SpMV kernel: 2D SELL SpMV
//
// Each CTA processes one slice of ROWS rows. The inner loop walks
// the slice's nonzeros COLS at a time. Because the SELL arrays are
// laid out column-major within a slice, the per-iteration load of
// 'sell_col_indices' and 'sell_values' is a contiguous load of
// ROWS*COLS contiguous elements — i.e. perfectly coalesced — even
// though the underlying rows have wildly different lengths.
//=============================================================================

template <int ROWS, int COLS, int OCCUPANCY = 8>
[[cutile::hint(0, occupancy = OCCUPANCY)]]
__tile_global__ void spmvSell(int num_rows,
                              const int* __restrict__ sell_col_indices,
                              const float* __restrict__ sell_values,
                              const int* __restrict__ slice_offsets,
                              const int* __restrict__ slice_widths,
                              const int* __restrict__ row_perm,
                              const float* __restrict__ vector_x,
                              float* __restrict__ vector_y) {
  using namespace ct::literals;

  using Tile2D = ct::tile<float, ct::shape<ROWS, COLS>>;
  using RowI = ct::tile<int, ct::shape<ROWS>>;
  using ColI = ct::tile<int, ct::shape<COLS>>;

  sell_col_indices = ct::assume_aligned<16>(sell_col_indices);
  sell_values = ct::assume_aligned<16>(sell_values);
  vector_x = ct::assume_aligned<16>(vector_x);

  int slice = ct::bid().x;
  int row_base = slice * ROWS;

  auto local_row = ct::iota<RowI>();
  auto row_valid = (row_base + local_row) < num_rows;
  /* destination row in y for each lane of the slice */
  auto actual_row = ct::load_masked(row_perm + row_base + local_row,
                                    row_valid, 0);

  int offset = slice_offsets[slice];
  int width = slice_widths[slice];

  /* Build 2D index tiles. row_2d broadcasts the in-slice row index
   * along the column dimension; col_base_2d broadcasts the COLS
   * lanes along the row dimension. The SELL element address is
   *   offset + (k + col) * ROWS + r
   * which becomes the body of the inner loop below. */
  auto row_2d = ct::broadcast(
      ct::reshape(local_row, ct::shape<ROWS, 1>{}), ct::shape<ROWS, COLS>{});
  auto col_base = ct::iota<ColI>();
  auto row_valid_2d = ct::broadcast(
      ct::reshape(row_valid, ct::shape<ROWS, 1>{}), ct::shape<ROWS, COLS>{});
  auto col_base_2d = ct::broadcast(
      ct::reshape(col_base, ct::shape<1, COLS>{}), ct::shape<ROWS, COLS>{});

  Tile2D acc = ct::zeros<Tile2D>();

  /* Loop-split: 'full_width' iterations need no per-element column
   * mask (every lane is within 'width'), and the optional trailing
   * iteration uses a mask. Eliminating the mask from the hot loop
   * saves a predicate evaluation per element per iteration. */
  int full_width = (width / COLS) * COLS;

  #pragma unroll 1
  for (int k = 0; k < full_width; k += COLS) {
    auto sell_idx = offset + (k + col_base_2d) * ROWS + row_2d;
    auto cols_2d = ct::load_masked(sell_col_indices + sell_idx,
                                   row_valid_2d, 0);
    auto vals_2d = ct::load_masked(sell_values + sell_idx,
                                   row_valid_2d, 0.0f);
    auto x_2d = ct::load_masked(vector_x + cols_2d, row_valid_2d, 0.0f);
    acc = acc + vals_2d * x_2d;
  }

  if (full_width < width) {
    auto col_offsets_2d = full_width + col_base_2d;
    auto valid = row_valid_2d & (col_offsets_2d < width);
    auto sell_idx = offset + col_offsets_2d * ROWS + row_2d;
    auto cols_2d = ct::load_masked(sell_col_indices + sell_idx, valid, 0);
    auto vals_2d = ct::load_masked(sell_values + sell_idx, valid, 0.0f);
    auto x_2d = ct::load_masked(vector_x + cols_2d, valid, 0.0f);
    acc = acc + vals_2d * x_2d;
  }

  /* Reduce along the column dimension to get one sum per slice row,
   * then scatter to the destination row in y. */
  auto row_sums = ct::sum(acc, 1_ic);
  ct::store_masked(vector_y + actual_row,
                   ct::reshape(row_sums, ct::shape<ROWS>{}), row_valid);
}

//=============================================================================
// Tile shape configuration
//
// The kernel is templated on (ROWS, COLS); ROWS is also the slice
// size in the SELL packing. We use a single shape sized for the
// random matrix generated below (~16 nonzeros per row on average).
//=============================================================================

constexpr int SLICE_ROWS = 64;
constexpr int TILE_COLS = 16;

//=============================================================================
// Sliced ELLPACK (SELL) matrix
//
// Layout:
//   slice s covers row_perm[s*SLICE_ROWS .. s*SLICE_ROWS + SLICE_ROWS)
//   slice_widths[s]  = max( nnz_per_row[row] for row in slice s )
//   slice_offsets[s] = sum_{t<s} slice_widths[t] * SLICE_ROWS
//   sell_col_indices[slice_offsets[s] + k * SLICE_ROWS + r] =
//       column index of the k-th nonzero of slice s, row r
//       (or 0 if that slot is padding)
//   sell_values[...] same but for the value
//=============================================================================

struct SellMatrix {
  int num_rows = 0;
  int num_cols = 0;
  int num_slices = 0;
  int nnz_total = 0;                  /* non-padding entries only */
  std::size_t total_sell_entries = 0; /* including padding */

  std::vector<int> row_perm;       /* size num_slices * SLICE_ROWS */
  std::vector<int> slice_offsets;  /* size num_slices */
  std::vector<int> slice_widths;   /* size num_slices */
  std::vector<int> row_lengths;    /* size num_slices * SLICE_ROWS, padded */
  std::vector<int> sell_col_indices;
  std::vector<float> sell_values;

  int* d_row_perm = nullptr;
  int* d_slice_offsets = nullptr;
  int* d_slice_widths = nullptr;
  int* d_sell_col_indices = nullptr;
  float* d_sell_values = nullptr;

  void uploadToDevice() {
    checkCudaErrors(cudaMalloc(&d_row_perm, row_perm.size() * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_slice_offsets,
                               slice_offsets.size() * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_slice_widths,
                               slice_widths.size() * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_sell_col_indices,
                               sell_col_indices.size() * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_sell_values,
                               sell_values.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_row_perm, row_perm.data(),
                               row_perm.size() * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_slice_offsets, slice_offsets.data(),
                               slice_offsets.size() * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_slice_widths, slice_widths.data(),
                               slice_widths.size() * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_sell_col_indices, sell_col_indices.data(),
                               sell_col_indices.size() * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_sell_values, sell_values.data(),
                               sell_values.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
  }

  void freeDevice() {
    if (d_row_perm) checkCudaErrors(cudaFree(d_row_perm));
    if (d_slice_offsets) checkCudaErrors(cudaFree(d_slice_offsets));
    if (d_slice_widths) checkCudaErrors(cudaFree(d_slice_widths));
    if (d_sell_col_indices) checkCudaErrors(cudaFree(d_sell_col_indices));
    if (d_sell_values) checkCudaErrors(cudaFree(d_sell_values));
    d_row_perm = nullptr;
    d_slice_offsets = nullptr;
    d_slice_widths = nullptr;
    d_sell_col_indices = nullptr;
    d_sell_values = nullptr;
  }
};

/* Build a SellMatrix from per-row column-index and value lists. Rows
 * are sorted by ascending length before slicing so that each slice's
 * longest row is close to its shortest — this minimizes the amount
 * of zero-padding required inside the slice. */
static SellMatrix packSell(int num_rows, int num_cols,
                           const std::vector<int>& row_lengths,
                           const std::vector<int>& row_cols_flat,
                           const std::vector<float>& row_vals_flat) {
  SellMatrix S;
  S.num_rows = num_rows;
  S.num_cols = num_cols;
  S.num_slices = (num_rows + SLICE_ROWS - 1) / SLICE_ROWS;

  /* prefix sums into the flattened row arrays */
  std::vector<int> prefix(num_rows + 1, 0);
  for (int r = 0; r < num_rows; ++r) {
    prefix[r + 1] = prefix[r] + row_lengths[r];
  }
  S.nnz_total = prefix[num_rows];

  /* sort row indices by ascending length */
  std::vector<int> perm(num_rows);
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&](int a, int b) {
    return row_lengths[a] < row_lengths[b];
  });

  /* pad the permutation up to a whole number of slices; the padding
   * slots map to row 0, but those lanes are masked out by row_valid
   * in the kernel and contribute nothing */
  std::size_t padded_rows =
      static_cast<std::size_t>(S.num_slices) * SLICE_ROWS;
  S.row_perm.assign(padded_rows, 0);
  S.row_lengths.assign(padded_rows, 0);
  for (int i = 0; i < num_rows; ++i) {
    S.row_perm[i] = perm[i];
    S.row_lengths[i] = row_lengths[perm[i]];
  }

  /* per-slice width = max row length within the slice */
  S.slice_widths.assign(S.num_slices, 0);
  for (int s = 0; s < S.num_slices; ++s) {
    int w = 0;
    for (int r = 0; r < SLICE_ROWS; ++r) {
      w = std::max(w, S.row_lengths[s * SLICE_ROWS + r]);
    }
    S.slice_widths[s] = w;
  }

  /* per-slice offset = prefix sum of slice_width * SLICE_ROWS */
  S.slice_offsets.assign(S.num_slices, 0);
  std::size_t running = 0;
  for (int s = 0; s < S.num_slices; ++s) {
    S.slice_offsets[s] = static_cast<int>(running);
    running += static_cast<std::size_t>(S.slice_widths[s]) * SLICE_ROWS;
  }
  S.total_sell_entries = running;

  /* pack into column-major slice layout */
  S.sell_col_indices.assign(S.total_sell_entries, 0);
  S.sell_values.assign(S.total_sell_entries, 0.0f);
  for (int s = 0; s < S.num_slices; ++s) {
    int offset = S.slice_offsets[s];
    for (int r = 0; r < SLICE_ROWS; ++r) {
      int global_idx = s * SLICE_ROWS + r;
      if (global_idx >= num_rows) continue;
      int row = S.row_perm[global_idx];
      int row_start = prefix[row];
      int row_len = row_lengths[row];
      for (int k = 0; k < row_len; ++k) {
        std::size_t dst = static_cast<std::size_t>(offset)
                          + static_cast<std::size_t>(k) * SLICE_ROWS + r;
        S.sell_col_indices[dst] = row_cols_flat[row_start + k];
        S.sell_values[dst] = row_vals_flat[row_start + k];
      }
      /* the remaining k in [row_len, slice_widths[s]) stays as the
       * zero-init padding written by the assign() calls above */
    }
  }

  return S;
}

//=============================================================================
// Random matrix generator (produces SellMatrix directly)
//
// Each row has a Poisson-distributed number of nonzeros; the column
// indices within a row are uniform random, then sorted and
// de-duplicated. packSell() handles the slice layout.
//=============================================================================

static SellMatrix generateRandom(int num_rows, int num_cols,
                                 int avg_nnz_per_row, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> col_dist(0, num_cols - 1);
  std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
  std::poisson_distribution<int> len_dist(static_cast<double>(avg_nnz_per_row));

  std::vector<int> cols_flat;
  std::vector<float> vals_flat;
  std::vector<int> row_lengths;
  row_lengths.reserve(num_rows);

  std::vector<int> scratch_cols;
  for (int r = 0; r < num_rows; ++r) {
    int len = std::min(num_cols, std::max(1, len_dist(rng)));
    scratch_cols.clear();
    scratch_cols.reserve(len);
    for (int k = 0; k < len; ++k) {
      scratch_cols.push_back(col_dist(rng));
    }
    std::sort(scratch_cols.begin(), scratch_cols.end());
    scratch_cols.erase(std::unique(scratch_cols.begin(), scratch_cols.end()),
                       scratch_cols.end());
    for (int c : scratch_cols) {
      cols_flat.push_back(c);
      vals_flat.push_back(val_dist(rng));
    }
    row_lengths.push_back(static_cast<int>(scratch_cols.size()));
  }

  return packSell(num_rows, num_cols, row_lengths, cols_flat, vals_flat);
}

//=============================================================================
// CPU reference SpMV — reads SELL directly so the sample has no
// dependency on CSR or any external sparse-matrix library.
//=============================================================================

static void cpuSpMV(const SellMatrix& S, const std::vector<float>& x,
                    std::vector<float>& y) {
  y.assign(S.num_rows, 0.0f);
  for (int s = 0; s < S.num_slices; ++s) {
    int offset = S.slice_offsets[s];
    int width = S.slice_widths[s];
    for (int r = 0; r < SLICE_ROWS; ++r) {
      int global_idx = s * SLICE_ROWS + r;
      if (global_idx >= S.num_rows) continue;
      int dst_row = S.row_perm[global_idx];
      int row_len = S.row_lengths[global_idx];
      float sum = 0.0f;
      for (int k = 0; k < row_len; ++k) {
        std::size_t src = static_cast<std::size_t>(offset)
                          + static_cast<std::size_t>(k) * SLICE_ROWS + r;
        sum += S.sell_values[src] * x[S.sell_col_indices[src]];
      }
      y[dst_row] = sum;
      (void)width;
    }
  }
}

/* Compare device result to the CPU reference. SpMV is performed in
 * single precision and the device kernel reduces in a different
 * order than the CPU reference, so we accept differences within a
 * relative tolerance OR a small absolute tolerance — whichever is
 * larger. */
static bool verify(const std::vector<float>& reference,
                   const std::vector<float>& result,
                   float rel_tol = 1e-2f, float abs_tol = 1e-4f) {
  float max_err = 0.0f;
  int bad_idx = -1;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    float diff = std::fabs(reference[i] - result[i]);
    float allowed = std::max(abs_tol, rel_tol * std::fabs(reference[i]));
    float over = diff - allowed;
    if (over > max_err) {
      max_err = over;
      bad_idx = static_cast<int>(i);
    }
  }
  if (max_err > 0.0f) {
    printf("Verification FAILED at index %d (ref=%g, got=%g, diff=%g)\n",
           bad_idx, reference[bad_idx], result[bad_idx],
           std::fabs(reference[bad_idx] - result[bad_idx]));
    return false;
  }
  return true;
}

//=============================================================================
// Main
//=============================================================================

int main() {
  /* Random sparse matrix: ~16 nonzeros per row on average, sized to
   * match the chosen tile shape (SLICE_ROWS = 64, TILE_COLS = 16). */
  SellMatrix S = generateRandom(/*num_rows=*/100000, /*num_cols=*/100000,
                                /*avg_nnz_per_row=*/16, /*seed=*/0xA5A5);

  printf("Random sparse matrix: rows=%d, cols=%d, nnz=%d, "
         "avg nnz/row=%.1f\n",
         S.num_rows, S.num_cols, S.nnz_total,
         static_cast<double>(S.nnz_total) / S.num_rows);
  printf("Tile configuration: ROWS=%d, COLS=%d (%d slices)\n",
         SLICE_ROWS, TILE_COLS, S.num_slices);

  /* host inputs */
  std::vector<float> h_x(S.num_cols);
  std::mt19937 rng(0xC0FFEE);
  std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
  for (float& v : h_x) v = x_dist(rng);

  /* CPU reference */
  std::vector<float> ref_y;
  cpuSpMV(S, h_x, ref_y);

  /* device allocations */
  S.uploadToDevice();
  float* d_x = nullptr;
  float* d_y = nullptr;
  checkCudaErrors(cudaMalloc(&d_x, S.num_cols * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y, S.num_rows * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_x, h_x.data(), S.num_cols * sizeof(float),
                             cudaMemcpyHostToDevice));

  /* Launch the SELL Tile kernel: one CTA per slice. */
  spmvSell<SLICE_ROWS, TILE_COLS><<<S.num_slices>>>(
      S.num_rows, S.d_sell_col_indices, S.d_sell_values,
      S.d_slice_offsets, S.d_slice_widths, S.d_row_perm, d_x, d_y);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  /* copy result back and verify */
  std::vector<float> h_y(S.num_rows);
  checkCudaErrors(cudaMemcpy(h_y.data(), d_y, S.num_rows * sizeof(float),
                             cudaMemcpyDeviceToHost));

  S.freeDevice();
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));

  if (!verify(ref_y, h_y)) {
    return 1;
  }

  printf("Success! Tile SpMV matches the CPU reference.\n");
  return 0;
}
