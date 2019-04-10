/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

// Utility function to extract unsigned chars from an
// unsigned integer

__device__ uchar4 int_to_uchar4(unsigned int in) {
  uchar4 bytes;
  bytes.x = in & 0x000000ff >> 0;
  bytes.y = in & 0x0000ff00 >> 8;
  bytes.z = in & 0x00ff0000 >> 16;
  bytes.w = in & 0xff000000 >> 24;
  return bytes;
}

// This function demonstrates some uses of the shuffle instruction
// in the generation of an integral image (also
// called a summed area table)
// The approach is two pass, a horizontal (scanline) then a vertical
// (column) pass.
// This is the horizontal pass kernel.
__global__ void shfl_intimage_rows(uint4 *img, uint4 *integral_image) {
  __shared__ int sums[128];

  int id = threadIdx.x;
  // pointer to head of current scanline
  uint4 *scanline = &img[blockIdx.x * 120];
  uint4 data;
  data = scanline[id];
  int result[16];
  int sum;
  unsigned int lane_id = id % warpSize;
  int warp_id = threadIdx.x / warpSize;

  uchar4 a = int_to_uchar4(data.x);
  uchar4 b = int_to_uchar4(data.y);
  uchar4 c = int_to_uchar4(data.z);
  uchar4 d = int_to_uchar4(data.w);

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

#pragma unroll

  for (int i = 4; i <= 7; i++) result[i] += result[3];

#pragma unroll

  for (int i = 8; i <= 11; i++) result[i] += result[7];

#pragma unroll

  for (int i = 12; i <= 15; i++) result[i] += result[11];

  sum = result[15];

  // the prefix sum for each thread's 16 value is computed,
  // now the final sums (result[15]) need to be shared
  // with the other threads and add.  To do this,
  // the __shfl_up() instruction is used and a shuffle scan
  // operation is performed to distribute the sums to the correct
  // threads
#pragma unroll

  for (int i = 1; i < 32; i *= 2) {
    unsigned int mask = 0xffffffff;
    int n = __shfl_up_sync(mask, sum, i, 32);

    if (lane_id >= i) {
#pragma unroll

      for (int i = 0; i < 16; i++) {
        result[i] += n;
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
  if (threadIdx.x % warpSize == warpSize - 1) {
    sums[warp_id] = result[15];
  }

  __syncthreads();

  if (warp_id == 0) {
    int warp_sum = sums[lane_id];
#pragma unroll

    for (int i = 1; i <= 32; i *= 2) {
      unsigned int mask = 0xffffffff;
      int n = __shfl_up_sync(mask, warp_sum, i, 32);

      if (lane_id >= i) warp_sum += n;
    }

    sums[lane_id] = warp_sum;
  }

  __syncthreads();

  int blockSum = 0;

  // fold in unused warp
  if (warp_id > 0) {
    blockSum = sums[warp_id - 1];
#pragma unroll

    for (int i = 0; i < 16; i++) {
      result[i] += blockSum;
    }
  }

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

  unsigned int mask = 0xffffffff;
  uint4 output;
  result[4] = __shfl_xor_sync(mask, result[4], 1, 32);
  result[5] = __shfl_xor_sync(mask, result[5], 1, 32);
  result[6] = __shfl_xor_sync(mask, result[6], 1, 32);
  result[7] = __shfl_xor_sync(mask, result[7], 1, 32);

  result[8] = __shfl_xor_sync(mask, result[8], 2, 32);
  result[9] = __shfl_xor_sync(mask, result[9], 2, 32);
  result[10] = __shfl_xor_sync(mask, result[10], 2, 32);
  result[11] = __shfl_xor_sync(mask, result[11], 2, 32);

  result[12] = __shfl_xor_sync(mask, result[12], 3, 32);
  result[13] = __shfl_xor_sync(mask, result[13], 3, 32);
  result[14] = __shfl_xor_sync(mask, result[14], 3, 32);
  result[15] = __shfl_xor_sync(mask, result[15], 3, 32);

  if (threadIdx.x % 4 == 0) {
    output = make_uint4(result[0], result[1], result[2], result[3]);
  }

  if (threadIdx.x % 4 == 1) {
    output = make_uint4(result[4], result[5], result[6], result[7]);
  }

  if (threadIdx.x % 4 == 2) {
    output = make_uint4(result[8], result[9], result[10], result[11]);
  }

  if (threadIdx.x % 4 == 3) {
    output = make_uint4(result[12], result[13], result[14], result[15]);
  }

  integral_image[blockIdx.x * 480 + threadIdx.x % 4 + (threadIdx.x / 4) * 16] =
      output;

  if (threadIdx.x % 4 == 2) {
    output = make_uint4(result[0], result[1], result[2], result[3]);
  }

  if (threadIdx.x % 4 == 3) {
    output = make_uint4(result[4], result[5], result[6], result[7]);
  }

  if (threadIdx.x % 4 == 0) {
    output = make_uint4(result[8], result[9], result[10], result[11]);
  }

  if (threadIdx.x % 4 == 1) {
    output = make_uint4(result[12], result[13], result[14], result[15]);
  }

  integral_image[blockIdx.x * 480 + (threadIdx.x + 2) % 4 +
                 (threadIdx.x / 4) * 16 + 8] = output;
  // continuing from the above example,
  // this use of __shfl_xor() places the y0..y3 and w0..w3 data
  // in order.
#pragma unroll

  for (int i = 0; i < 16; i++) {
    result[i] = __shfl_xor_sync(mask, result[i], 1, 32);
  }

  if (threadIdx.x % 4 == 0) {
    output = make_uint4(result[0], result[1], result[2], result[3]);
  }

  if (threadIdx.x % 4 == 1) {
    output = make_uint4(result[4], result[5], result[6], result[7]);
  }

  if (threadIdx.x % 4 == 2) {
    output = make_uint4(result[8], result[9], result[10], result[11]);
  }

  if (threadIdx.x % 4 == 3) {
    output = make_uint4(result[12], result[13], result[14], result[15]);
  }

  integral_image[blockIdx.x * 480 + threadIdx.x % 4 + (threadIdx.x / 4) * 16 +
                 4] = output;

  if (threadIdx.x % 4 == 2) {
    output = make_uint4(result[0], result[1], result[2], result[3]);
  }

  if (threadIdx.x % 4 == 3) {
    output = make_uint4(result[4], result[5], result[6], result[7]);
  }

  if (threadIdx.x % 4 == 0) {
    output = make_uint4(result[8], result[9], result[10], result[11]);
  }

  if (threadIdx.x % 4 == 1) {
    output = make_uint4(result[12], result[13], result[14], result[15]);
  }

  integral_image[blockIdx.x * 480 + (threadIdx.x + 2) % 4 +
                 (threadIdx.x / 4) * 16 + 12] = output;
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
