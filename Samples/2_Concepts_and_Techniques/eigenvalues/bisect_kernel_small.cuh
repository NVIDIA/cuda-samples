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

/* Determine eigenvalues for small symmetric, tridiagonal matrix */

#ifndef _BISECT_KERNEL_SMALL_H_
#define _BISECT_KERNEL_SMALL_H_

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// includes, project
#include "config.h"
#include "util.h"

// additional kernel
#include "bisect_util.cu"

////////////////////////////////////////////////////////////////////////////////
//! Bisection to find eigenvalues of a real, symmetric, and tridiagonal matrix
//! @param  g_d  diagonal elements in global memory
//! @param  g_s  superdiagonal elements in global elements (stored so that the
//!              element *(g_s - 1) can be accessed an equals 0
//! @param  n   size of matrix
//! @param  lg  lower bound of input interval (e.g. Gerschgorin interval)
//! @param  ug  upper bound of input interval (e.g. Gerschgorin interval)
//! @param  lg_eig_count  number of eigenvalues that are smaller than \a lg
//! @param  lu_eig_count  number of eigenvalues that are smaller than \a lu
//! @param  epsilon  desired accuracy of eigenvalues to compute
////////////////////////////////////////////////////////////////////////////////
__global__ void bisectKernel(float *g_d, float *g_s, const unsigned int n,
                             float *g_left, float *g_right,
                             unsigned int *g_left_count,
                             unsigned int *g_right_count, const float lg,
                             const float ug, const unsigned int lg_eig_count,
                             const unsigned int ug_eig_count, float epsilon) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // intervals (store left and right because the subdivision tree is in general
  // not dense
  __shared__ float s_left[MAX_THREADS_BLOCK_SMALL_MATRIX];
  __shared__ float s_right[MAX_THREADS_BLOCK_SMALL_MATRIX];

  // number of eigenvalues that are smaller than s_left / s_right
  // (correspondence is realized via indices)
  __shared__ unsigned int s_left_count[MAX_THREADS_BLOCK_SMALL_MATRIX];
  __shared__ unsigned int s_right_count[MAX_THREADS_BLOCK_SMALL_MATRIX];

  // helper for stream compaction
  __shared__ unsigned int s_compaction_list[MAX_THREADS_BLOCK_SMALL_MATRIX + 1];

  // state variables for whole block
  // if 0 then compaction of second chunk of child intervals is not necessary
  // (because all intervals had exactly one non-dead child)
  __shared__ unsigned int compact_second_chunk;
  __shared__ unsigned int all_threads_converged;

  // number of currently active threads
  __shared__ unsigned int num_threads_active;

  // number of threads to use for stream compaction
  __shared__ unsigned int num_threads_compaction;

  // helper for exclusive scan
  unsigned int *s_compaction_list_exc = s_compaction_list + 1;

  // variables for currently processed interval
  // left and right limit of active interval
  float left = 0.0f;
  float right = 0.0f;
  unsigned int left_count = 0;
  unsigned int right_count = 0;
  // midpoint of active interval
  float mid = 0.0f;
  // number of eigenvalues smaller then mid
  unsigned int mid_count = 0;
  // affected from compaction
  unsigned int is_active_second = 0;

  s_compaction_list[threadIdx.x] = 0;
  s_left[threadIdx.x] = 0;
  s_right[threadIdx.x] = 0;
  s_left_count[threadIdx.x] = 0;
  s_right_count[threadIdx.x] = 0;

  cg::sync(cta);

  // set up initial configuration
  if (0 == threadIdx.x) {
    s_left[0] = lg;
    s_right[0] = ug;
    s_left_count[0] = lg_eig_count;
    s_right_count[0] = ug_eig_count;

    compact_second_chunk = 0;
    num_threads_active = 1;

    num_threads_compaction = 1;
  }

  // for all active threads read intervals from the last level
  // the number of (worst case) active threads per level l is 2^l
  while (true) {
    all_threads_converged = 1;
    cg::sync(cta);

    is_active_second = 0;
    subdivideActiveInterval(threadIdx.x, s_left, s_right, s_left_count,
                            s_right_count, num_threads_active, left, right,
                            left_count, right_count, mid,
                            all_threads_converged);

    cg::sync(cta);

    // check if done
    if (1 == all_threads_converged) {
      break;
    }

    cg::sync(cta);

    // compute number of eigenvalues smaller than mid
    // use all threads for reading the necessary matrix data from global
    // memory
    // use s_left and s_right as scratch space for diagonal and
    // superdiagonal of matrix
    mid_count = computeNumSmallerEigenvals(g_d, g_s, n, mid, threadIdx.x,
                                           num_threads_active, s_left, s_right,
                                           (left == right), cta);

    cg::sync(cta);

    // store intervals
    // for all threads store the first child interval in a continuous chunk of
    // memory, and the second child interval -- if it exists -- in a second
    // chunk; it is likely that all threads reach convergence up to
    // \a epsilon at the same level; furthermore, for higher level most / all
    // threads will have only one child, storing the first child compactly will
    // (first) avoid to perform a compaction step on the first chunk, (second)
    // make it for higher levels (when all threads / intervals have
    // exactly one child)  unnecessary to perform a compaction of the second
    // chunk
    if (threadIdx.x < num_threads_active) {
      if (left != right) {
        // store intervals
        storeNonEmptyIntervals(threadIdx.x, num_threads_active, s_left, s_right,
                               s_left_count, s_right_count, left, mid, right,
                               left_count, mid_count, right_count, epsilon,
                               compact_second_chunk, s_compaction_list_exc,
                               is_active_second);
      } else {
        storeIntervalConverged(
            s_left, s_right, s_left_count, s_right_count, left, mid, right,
            left_count, mid_count, right_count, s_compaction_list_exc,
            compact_second_chunk, num_threads_active, is_active_second);
      }
    }

    // necessary so that compact_second_chunk is up-to-date
    cg::sync(cta);

    // perform compaction of chunk where second children are stored
    // scan of (num_threads_active / 2) elements, thus at most
    // (num_threads_active / 4) threads are needed
    if (compact_second_chunk > 0) {
      createIndicesCompaction(s_compaction_list_exc, num_threads_compaction,
                              cta);

      compactIntervals(s_left, s_right, s_left_count, s_right_count, mid, right,
                       mid_count, right_count, s_compaction_list,
                       num_threads_active, is_active_second);
    }

    cg::sync(cta);

    if (0 == threadIdx.x) {
      // update number of active threads with result of reduction
      num_threads_active += s_compaction_list[num_threads_active];

      num_threads_compaction = ceilPow2(num_threads_active);

      compact_second_chunk = 0;
    }

    cg::sync(cta);
  }

  cg::sync(cta);

  // write resulting intervals to global mem
  // for all threads write if they have been converged to an eigenvalue to
  // a separate array

  // at most n valid intervals
  if (threadIdx.x < n) {
    // intervals converged so left and right limit are identical
    g_left[threadIdx.x] = s_left[threadIdx.x];
    // left count is sufficient to have global order
    g_left_count[threadIdx.x] = s_left_count[threadIdx.x];
  }
}

#endif  // #ifndef _BISECT_KERNEL_SMALL_H_
