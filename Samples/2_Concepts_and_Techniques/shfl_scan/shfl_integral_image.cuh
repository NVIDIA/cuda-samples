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
 
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Utility function to extract unsigned chars from an
// unsigned integer
__device__ uchar4 uint_to_uchar4(const unsigned int in) {
  return make_uchar4((in & 0x000000ff) >> 0, (in & 0x0000ff00) >> 8,
                     (in & 0x00ff0000) >> 16, (in & 0xff000000) >> 24);
}

// Utility for dealing with vector data at different levels.
struct packed_result {
  uint4 x, y, z, w;
};

__device__ packed_result get_prefix_sum(const uint4 &data,
                                        const cg::thread_block &cta) {
  const auto tile = cg::tiled_partition<32>(cta);

  __shared__ unsigned int sums[128];
  const unsigned int lane_id = tile.thread_rank();
  const unsigned int warp_id = tile.meta_group_rank();

  unsigned int result[16] = {};
  {
    const uchar4 a = uint_to_uchar4(data.x);
    const uchar4 b = uint_to_uchar4(data.y);
    const uchar4 c = uint_to_uchar4(data.z);
    const uchar4 d = uint_to_uchar4(data.w);

    result[0] = a.x;
    result[1] = a.x + a.y;
    result[2] = a.x + a.y + a.z;
    result[3] = a.x + a.y + a.z + a.w;

    result[4] = b.x;
    result[5] = b.x + b.y;
    result[6] = b.x + b.y + b.z;
    result[7] = b.x + b.y + b.z + b.w;

    result[8] = c.x;
    result[9] = c.x + c.y;
    result[10] = c.x + c.y + c.z;
    result[11] = c.x + c.y + c.z + c.w;

    result[12] = d.x;
    result[13] = d.x + d.y;
    result[14] = d.x + d.y + d.z;
    result[15] = d.x + d.y + d.z + d.w;
  }

#pragma unroll
  for (unsigned int i = 4; i <= 7; i++) result[i] += result[3];

#pragma unroll
  for (unsigned int i = 8; i <= 11; i++) result[i] += result[7];

#pragma unroll
  for (unsigned int i = 12; i <= 15; i++) result[i] += result[11];

  unsigned int sum = result[15];

  // the prefix sum for each thread's 16 value is computed,
  // now the final sums (result[15]) need to be shared
  // with the other threads and add.  To do this,
  // the __shfl_up() instruction is used and a shuffle scan
  // operation is performed to distribute the sums to the correct
  // threads

#pragma unroll
  for (unsigned int i = 1; i < 32; i *= 2) {
    const unsigned int n = tile.shfl_up(sum, i);

    if (lane_id >= i) {
#pragma unroll
      for (unsigned int j = 0; j < 16; j++) {
        result[j] += n;
      }

      sum += n;
    }
  }

  // Now the final sum for the warp must be shared
  // between warps.  This is done by each warp
  // having a thread store to shared memory, then
  // having some other warp load the values and
  // compute a prefix sum, again by using __shfl_up.
  // The results are uniformly added back to the warps.
  // last thread in the warp holding sum of the warp
  // places that in shared
  if (tile.thread_rank() == (tile.size() - 1)) {
    sums[warp_id] = result[15];
  }

  __syncthreads();

  if (warp_id == 0) {
    unsigned int warp_sum = sums[lane_id];

#pragma unroll
    for (unsigned int i = 1; i <= 16; i *= 2) {
      const unsigned int n = tile.shfl_up(warp_sum, i);

      if (lane_id >= i) warp_sum += n;
    }

    sums[lane_id] = warp_sum;
  }

  __syncthreads();

  // fold in unused warp
  if (warp_id > 0) {
    const unsigned int blockSum = sums[warp_id - 1];

#pragma unroll
    for (unsigned int i = 0; i < 16; i++) {
      result[i] += blockSum;
    }
  }

  packed_result out;
  memcpy(&out, result, sizeof(out));
  return out;
}

// This function demonstrates some uses of the shuffle instruction
// in the generation of an integral image (also
// called a summed area table)
// The approach is two pass, a horizontal (scanline) then a vertical
// (column) pass.
// This is the horizontal pass kernel.
__global__ void shfl_intimage_rows(const uint4 *img, uint4 *integral_image) {
  const auto cta = cg::this_thread_block();
  const auto tile = cg::tiled_partition<32>(cta);

  const unsigned int id = threadIdx.x;
  // pointer to head of current scanline
  const uint4 *scanline = &img[blockIdx.x * 120];
  packed_result result = get_prefix_sum(scanline[id], cta);

  // This access helper allows packed_result to stay optimized as registers
  // rather than spill to stack
  auto idxToElem = [&result](unsigned int idx) -> const uint4 {
    switch (idx) {
      case 0:
        return result.x;
      case 1:
        return result.y;
      case 2:
        return result.z;
      case 3:
        return result.w;
    }
    return {};
  };

  // assemble result
  // Each thread has 16 values to write, which are
  // now integer data (to avoid overflow).  Instead of
  // each thread writing consecutive uint4s, the
  // approach shown here experiments using
  // the shuffle command to reformat the data
  // inside the registers so that each thread holds
  // consecutive data to be written so larger contiguous
  // segments can be assembled for writing.
  /*
    For example data that needs to be written as

    GMEM[16] <- x0 x1 x2 x3 y0 y1 y2 y3 z0 z1 z2 z3 w0 w1 w2 w3
    but is stored in registers (r0..r3), in four threads (0..3) as:

    threadId   0  1  2  3
      r0      x0 y0 z0 w0
      r1      x1 y1 z1 w1
      r2      x2 y2 z2 w2
      r3      x3 y3 z3 w3

      after apply __shfl_xor operations to move data between registers r1..r3:

    threadId  00 01 10 11
              x0 y0 z0 w0
     xor(01)->y1 x1 w1 z1
     xor(10)->z2 w2 x2 y2
     xor(11)->w3 z3 y3 x3

     and now x0..x3, and z0..z3 can be written out in order by all threads.

     In the current code, each register above is actually representing
     four integers to be written as uint4's to GMEM.
  */

  const unsigned int idMask = id & 3;
  const unsigned int idSwizzle = (id + 2) & 3;
  const unsigned int idShift = (id >> 2) << 4;
  const unsigned int blockOffset = blockIdx.x * 480;

  // Use CG tile to warp shuffle vector types
  result.y = tile.shfl_xor(result.y, 1);
  result.z = tile.shfl_xor(result.z, 2);
  result.w = tile.shfl_xor(result.w, 3);

  // First batch
  integral_image[blockOffset + idMask + idShift] = idxToElem(idMask);
  // Second batch offset by 2
  integral_image[blockOffset + idSwizzle + idShift + 8] = idxToElem(idSwizzle);

  // continuing from the above example,
  // this use of __shfl_xor() places the y0..y3 and w0..w3 data
  // in order.
  result.x = tile.shfl_xor(result.x, 1);
  result.y = tile.shfl_xor(result.y, 1);
  result.z = tile.shfl_xor(result.z, 1);
  result.w = tile.shfl_xor(result.w, 1);

  // First batch
  integral_image[blockOffset + idMask + idShift + 4] = idxToElem(idMask);
  // Second batch offset by 2
  integral_image[blockOffset + idSwizzle + idShift + 12] = idxToElem(idSwizzle);
}

// This kernel computes columnwise prefix sums.  When the data input is
// the row sums from above, this completes the integral image.
// The approach here is to have each block compute a local set of sums.
// First , the data covered by the block is loaded into shared memory,
// then instead of performing a sum in shared memory using __syncthreads
// between stages, the data is reformatted so that the necessary sums
// occur inside warps and the shuffle scan operation is used.
// The final set of sums from the block is then propagated, with the block
// computing "down" the image and adding the running sum to the local
// block sums.
__global__ void shfl_vertical_shfl(unsigned int *img, int width, int height) {
  __shared__ unsigned int sums[32][9];
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  // int warp_id = threadIdx.x / warpSize ;
  unsigned int lane_id = tidx % 8;
  // int rows_per_thread = (height / blockDim. y) ;
  // int start_row = rows_per_thread * threadIdx.y;
  unsigned int stepSum = 0;
  unsigned int mask = 0xffffffff;

  sums[threadIdx.x][threadIdx.y] = 0;
  __syncthreads();

  for (int step = 0; step < 135; step++) {
    unsigned int sum = 0;
    unsigned int *p = img + (threadIdx.y + step * 8) * width + tidx;

    sum = *p;
    sums[threadIdx.x][threadIdx.y] = sum;
    __syncthreads();

    // place into SMEM
    // shfl scan reduce the SMEM, reformating so the column
    // sums are computed in a warp
    // then read out properly
    int partial_sum = 0;
    int j = threadIdx.x % 8;
    int k = threadIdx.x / 8 + threadIdx.y * 4;

    partial_sum = sums[k][j];

    for (int i = 1; i <= 8; i *= 2) {
      int n = __shfl_up_sync(mask, partial_sum, i, 32);

      if (lane_id >= i) partial_sum += n;
    }

    sums[k][j] = partial_sum;
    __syncthreads();

    if (threadIdx.y > 0) {
      sum += sums[threadIdx.x][threadIdx.y - 1];
    }

    sum += stepSum;
    stepSum += sums[threadIdx.x][blockDim.y - 1];
    __syncthreads();
    *p = sum;
  }
}
