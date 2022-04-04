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

/* Perform second step of bisection algorithm for large matrices for
 * intervals that contained after the first step more than one eigenvalue
 */

#ifndef _BISECT_KERNEL_LARGE_MULTI_H_
#define _BISECT_KERNEL_LARGE_MULTI_H_

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
// includes, project
#include "config.h"
#include "util.h"

// additional kernel
#include "bisect_util.cu"

////////////////////////////////////////////////////////////////////////////////
//! Perform second step of bisection algorithm for large matrices for
//! intervals that after the first step contained more than one eigenvalue
//! @param  g_d  diagonal elements of symmetric, tridiagonal matrix
//! @param  g_s  superdiagonal elements of symmetric, tridiagonal matrix
//! @param  n    matrix size
//! @param  blocks_mult  start addresses of blocks of intervals that are
//!                      processed by one block of threads, each of the
//!                      intervals contains more than one eigenvalue
//! @param  blocks_mult_sum  total number of eigenvalues / singleton intervals
//!                          in one block of intervals
//! @param  g_left  left limits of intervals
//! @param  g_right  right limits of intervals
//! @param  g_left_count  number of eigenvalues less than left limits
//! @param  g_right_count  number of eigenvalues less than right limits
//! @param  g_lambda  final eigenvalue
//! @param  g_pos  index of eigenvalue (in ascending order)
//! @param  precision  desired precision of eigenvalues
////////////////////////////////////////////////////////////////////////////////
__global__ void bisectKernelLarge_MultIntervals(
    float *g_d, float *g_s, const unsigned int n, unsigned int *blocks_mult,
    unsigned int *blocks_mult_sum, float *g_left, float *g_right,
    unsigned int *g_left_count, unsigned int *g_right_count, float *g_lambda,
    unsigned int *g_pos, float precision) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  const unsigned int tid = threadIdx.x;

  // left and right limits of interval
  __shared__ float s_left[2 * MAX_THREADS_BLOCK];
  __shared__ float s_right[2 * MAX_THREADS_BLOCK];

  // number of eigenvalues smaller than interval limits
  __shared__ unsigned int s_left_count[2 * MAX_THREADS_BLOCK];
  __shared__ unsigned int s_right_count[2 * MAX_THREADS_BLOCK];

  // helper array for chunk compaction of second chunk
  __shared__ unsigned int s_compaction_list[2 * MAX_THREADS_BLOCK + 1];
  // compaction list helper for exclusive scan
  unsigned int *s_compaction_list_exc = s_compaction_list + 1;

  // flag if all threads are converged
  __shared__ unsigned int all_threads_converged;
  // number of active threads
  __shared__ unsigned int num_threads_active;
  // number of threads to employ for compaction
  __shared__ unsigned int num_threads_compaction;
  // flag if second chunk has to be compacted
  __shared__ unsigned int compact_second_chunk;

  // parameters of block of intervals processed by this block of threads
  __shared__ unsigned int c_block_start;
  __shared__ unsigned int c_block_end;
  __shared__ unsigned int c_block_offset_output;

  // midpoint of currently active interval of the thread
  float mid = 0.0f;
  // number of eigenvalues smaller than \a mid
  unsigned int mid_count = 0;
  // current interval parameter
  float left;
  float right;
  unsigned int left_count;
  unsigned int right_count;
  // helper for compaction, keep track which threads have a second child
  unsigned int is_active_second = 0;

  // initialize common start conditions
  if (0 == tid) {
    c_block_start = blocks_mult[blockIdx.x];
    c_block_end = blocks_mult[blockIdx.x + 1];
    c_block_offset_output = blocks_mult_sum[blockIdx.x];

    num_threads_active = c_block_end - c_block_start;
    s_compaction_list[0] = 0;
    num_threads_compaction = ceilPow2(num_threads_active);

    all_threads_converged = 1;
    compact_second_chunk = 0;
  }

  cg::sync(cta);

  // read data into shared memory
  if (tid < num_threads_active) {
    s_left[tid] = g_left[c_block_start + tid];
    s_right[tid] = g_right[c_block_start + tid];
    s_left_count[tid] = g_left_count[c_block_start + tid];
    s_right_count[tid] = g_right_count[c_block_start + tid];
  }

  cg::sync(cta);

  // do until all threads converged
  while (true) {
    // for (int iter=0; iter < 0; iter++) {

    // subdivide interval if currently active and not already converged
    subdivideActiveInterval(tid, s_left, s_right, s_left_count, s_right_count,
                            num_threads_active, left, right, left_count,
                            right_count, mid, all_threads_converged);

    cg::sync(cta);

    // stop if all eigenvalues have been found
    if (1 == all_threads_converged) {
      break;
    }

    // compute number of eigenvalues smaller than mid for active and not
    // converged intervals, use all threads for loading data from gmem and
    // s_left and s_right as scratch space to store the data load from gmem
    // in shared memory
    mid_count = computeNumSmallerEigenvalsLarge(g_d, g_s, n, mid, tid,
                                                num_threads_active, s_left,
                                                s_right, (left == right), cta);

    cg::sync(cta);

    if (tid < num_threads_active) {
      // store intervals
      if (left != right) {
        storeNonEmptyIntervals(tid, num_threads_active, s_left, s_right,
                               s_left_count, s_right_count, left, mid, right,
                               left_count, mid_count, right_count, precision,
                               compact_second_chunk, s_compaction_list_exc,
                               is_active_second);
      } else {
        storeIntervalConverged(
            s_left, s_right, s_left_count, s_right_count, left, mid, right,
            left_count, mid_count, right_count, s_compaction_list_exc,
            compact_second_chunk, num_threads_active, is_active_second);
      }
    }

    cg::sync(cta);

    // compact second chunk of intervals if any of the threads generated
    // two child intervals
    if (1 == compact_second_chunk) {
      createIndicesCompaction(s_compaction_list_exc, num_threads_compaction,
                              cta);

      compactIntervals(s_left, s_right, s_left_count, s_right_count, mid, right,
                       mid_count, right_count, s_compaction_list,
                       num_threads_active, is_active_second);
    }

    cg::sync(cta);

    // update state variables
    if (0 == tid) {
      num_threads_active += s_compaction_list[num_threads_active];
      num_threads_compaction = ceilPow2(num_threads_active);

      compact_second_chunk = 0;
      all_threads_converged = 1;
    }

    cg::sync(cta);

    // clear
    s_compaction_list_exc[threadIdx.x] = 0;
    s_compaction_list_exc[threadIdx.x + blockDim.x] = 0;

    cg::sync(cta);

  }  // end until all threads converged

  // write data back to global memory
  if (tid < num_threads_active) {
    unsigned int addr = c_block_offset_output + tid;

    g_lambda[addr] = s_left[tid];
    g_pos[addr] = s_right_count[tid];
  }
}

#endif  // #ifndef _BISECT_KERNEL_LARGE_MULTI_H_
