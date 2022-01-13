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

/* Utility / shared functionality for bisection kernels */

#ifndef _BISECT_UTIL_H_
#define _BISECT_UTIL_H_

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// includes, project
#include "config.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////
//! Compute the next lower power of two of n
//! @param  n  number for which next higher power of two is sought
////////////////////////////////////////////////////////////////////////////////
__device__ inline int floorPow2(int n) {
  // early out if already power of two
  if (0 == (n & (n - 1))) {
    return n;
  }

  int exp;
  frexp((float)n, &exp);
  return (1 << (exp - 1));
}

////////////////////////////////////////////////////////////////////////////////
//! Compute the next higher power of two of n
//! @param  n  number for which next higher power of two is sought
////////////////////////////////////////////////////////////////////////////////
__device__ inline int ceilPow2(int n) {
  // early out if already power of two
  if (0 == (n & (n - 1))) {
    return n;
  }

  int exp;
  frexp((float)n, &exp);
  return (1 << exp);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute midpoint of interval [\a left, \a right] avoiding overflow if
//! possible
//! @param left   left / lower limit of interval
//! @param right  right / upper limit of interval
////////////////////////////////////////////////////////////////////////////////
__device__ inline float computeMidpoint(const float left, const float right) {
  float mid;

  if (sign_f(left) == sign_f(right)) {
    mid = left + (right - left) * 0.5f;
  } else {
    mid = (left + right) * 0.5f;
  }

  return mid;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if interval converged and store appropriately
//! @param  addr    address where to store the information of the interval
//! @param  s_left  shared memory storage for left interval limits
//! @param  s_right  shared memory storage for right interval limits
//! @param  s_left_count  shared memory storage for number of eigenvalues less
//!                       than left interval limits
//! @param  s_right_count  shared memory storage for number of eigenvalues less
//!                       than right interval limits
//! @param  left   lower limit of interval
//! @param  right  upper limit of interval
//! @param  left_count  eigenvalues less than \a left
//! @param  right_count  eigenvalues less than \a right
//! @param  precision  desired precision for eigenvalues
////////////////////////////////////////////////////////////////////////////////
template <class S, class T>
__device__ void storeInterval(unsigned int addr, float *s_left, float *s_right,
                              T *s_left_count, T *s_right_count, float left,
                              float right, S left_count, S right_count,
                              float precision) {
  s_left_count[addr] = left_count;
  s_right_count[addr] = right_count;

  // check if interval converged
  float t0 = abs(right - left);
  float t1 = max(abs(left), abs(right)) * precision;

  if (t0 <= max(MIN_ABS_INTERVAL, t1)) {
    // compute mid point
    float lambda = computeMidpoint(left, right);

    // mark as converged
    s_left[addr] = lambda;
    s_right[addr] = lambda;
  } else {
    // store current limits
    s_left[addr] = left;
    s_right[addr] = right;
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute number of eigenvalues that are smaller than x given a symmetric,
//! real, and tridiagonal matrix
//! @param  g_d  diagonal elements stored in global memory
//! @param  g_s  superdiagonal elements stored in global memory
//! @param  n    size of matrix
//! @param  x    value for which the number of eigenvalues that are smaller is
//!              seeked
//! @param  tid  thread identified (e.g. threadIdx.x or gtid)
//! @param  num_intervals_active  number of active intervals / threads that
//!                               currently process an interval
//! @param  s_d  scratch space to store diagonal entries of the tridiagonal
//!              matrix in shared memory
//! @param  s_s  scratch space to store superdiagonal entries of the tridiagonal
//!              matrix in shared memory
//! @param  converged  flag if the current thread is already converged (that
//!         is count does not have to be computed)
////////////////////////////////////////////////////////////////////////////////
__device__ inline unsigned int computeNumSmallerEigenvals(
    float *g_d, float *g_s, const unsigned int n, const float x,
    const unsigned int tid, const unsigned int num_intervals_active, float *s_d,
    float *s_s, unsigned int converged, cg::thread_block cta) {
  float delta = 1.0f;
  unsigned int count = 0;

  cg::sync(cta);

  // read data into shared memory
  if (threadIdx.x < n) {
    s_d[threadIdx.x] = *(g_d + threadIdx.x);
    s_s[threadIdx.x] = *(g_s + threadIdx.x - 1);
  }

  cg::sync(cta);

  // perform loop only for active threads
  if ((tid < num_intervals_active) && (0 == converged)) {
    // perform (optimized) Gaussian elimination to determine the number
    // of eigenvalues that are smaller than n
    for (unsigned int k = 0; k < n; ++k) {
      delta = s_d[k] - x - (s_s[k] * s_s[k]) / delta;
      count += (delta < 0) ? 1 : 0;
    }

  }  // end if thread currently processing an interval

  return count;
}

////////////////////////////////////////////////////////////////////////////////
//! Compute number of eigenvalues that are smaller than x given a symmetric,
//! real, and tridiagonal matrix
//! @param  g_d  diagonal elements stored in global memory
//! @param  g_s  superdiagonal elements stored in global memory
//! @param  n    size of matrix
//! @param  x    value for which the number of eigenvalues that are smaller is
//!              seeked
//! @param  tid  thread identified (e.g. threadIdx.x or gtid)
//! @param  num_intervals_active  number of active intervals / threads that
//!                               currently process an interval
//! @param  s_d  scratch space to store diagonal entries of the tridiagonal
//!              matrix in shared memory
//! @param  s_s  scratch space to store superdiagonal entries of the tridiagonal
//!              matrix in shared memory
//! @param  converged  flag if the current thread is already converged (that
//!         is count does not have to be computed)
////////////////////////////////////////////////////////////////////////////////
__device__ inline unsigned int computeNumSmallerEigenvalsLarge(
    float *g_d, float *g_s, const unsigned int n, const float x,
    const unsigned int tid, const unsigned int num_intervals_active, float *s_d,
    float *s_s, unsigned int converged, cg::thread_block cta) {
  float delta = 1.0f;
  unsigned int count = 0;

  unsigned int rem = n;

  // do until whole diagonal and superdiagonal has been loaded and processed
  for (unsigned int i = 0; i < n; i += blockDim.x) {
    cg::sync(cta);

    // read new chunk of data into shared memory
    if ((i + threadIdx.x) < n) {
      s_d[threadIdx.x] = *(g_d + i + threadIdx.x);
      s_s[threadIdx.x] = *(g_s + i + threadIdx.x - 1);
    }

    cg::sync(cta);

    if (tid < num_intervals_active) {
      // perform (optimized) Gaussian elimination to determine the number
      // of eigenvalues that are smaller than n
      for (unsigned int k = 0; k < min(rem, blockDim.x); ++k) {
        delta = s_d[k] - x - (s_s[k] * s_s[k]) / delta;
        // delta = (abs( delta) < (1.0e-10)) ? -(1.0e-10) : delta;
        count += (delta < 0) ? 1 : 0;
      }

    }  // end if thread currently processing an interval

    rem -= blockDim.x;
  }

  return count;
}

////////////////////////////////////////////////////////////////////////////////
//! Store all non-empty intervals resulting from the subdivision of the interval
//! currently processed by the thread
//! @param  addr  base address for storing intervals
//! @param  num_threads_active  number of threads / intervals in current sweep
//! @param  s_left  shared memory storage for left interval limits
//! @param  s_right  shared memory storage for right interval limits
//! @param  s_left_count  shared memory storage for number of eigenvalues less
//!                       than left interval limits
//! @param  s_right_count  shared memory storage for number of eigenvalues less
//!                       than right interval limits
//! @param  left   lower limit of interval
//! @param  mid    midpoint of interval
//! @param  right  upper limit of interval
//! @param  left_count  eigenvalues less than \a left
//! @param  mid_count  eigenvalues less than \a mid
//! @param  right_count  eigenvalues less than \a right
//! @param  precision  desired precision for eigenvalues
//! @param  compact_second_chunk  shared mem flag if second chunk is used and
//!                               ergo requires compaction
//! @param  s_compaction_list_exc  helper array for stream compaction,
//!                                s_compaction_list_exc[tid] = 1 when the
//!                                thread generated two child intervals
//! @is_active_interval  mark is thread has a second non-empty child interval
////////////////////////////////////////////////////////////////////////////////
template <class S, class T>
__device__ void storeNonEmptyIntervals(
    unsigned int addr, const unsigned int num_threads_active, float *s_left,
    float *s_right, T *s_left_count, T *s_right_count, float left, float mid,
    float right, const S left_count, const S mid_count, const S right_count,
    float precision, unsigned int &compact_second_chunk,
    T *s_compaction_list_exc, unsigned int &is_active_second) {
  // check if both child intervals are valid
  if ((left_count != mid_count) && (mid_count != right_count)) {
    // store the left interval
    storeInterval(addr, s_left, s_right, s_left_count, s_right_count, left, mid,
                  left_count, mid_count, precision);

    // mark that a second interval has been generated, only stored after
    // stream compaction of second chunk
    is_active_second = 1;
    s_compaction_list_exc[threadIdx.x] = 1;
    atomicExch(&compact_second_chunk, 1);
  } else {
    // only one non-empty child interval

    // mark that no second child
    is_active_second = 0;
    s_compaction_list_exc[threadIdx.x] = 0;

    // store the one valid child interval
    if (left_count != mid_count) {
      storeInterval(addr, s_left, s_right, s_left_count, s_right_count, left,
                    mid, left_count, mid_count, precision);
    } else {
      storeInterval(addr, s_left, s_right, s_left_count, s_right_count, mid,
                    right, mid_count, right_count, precision);
    }
  }
}
////////////////////////////////////////////////////////////////////////////////
//! Create indices for compaction, that is process \a s_compaction_list_exc
//! which is 1 for intervals that generated a second child and 0 otherwise
//! and create for each of the non-zero elements the index where the new
//! interval belongs to in a compact representation of all generated second
//! childs
//! @param   s_compaction_list_exc  list containing the flags which threads
//!                                 generated two children
//! @param   num_threads_compaction number of threads to employ for compaction
////////////////////////////////////////////////////////////////////////////////
template <class T>
__device__ void createIndicesCompaction(T *s_compaction_list_exc,
                                        unsigned int num_threads_compaction,
                                        cg::thread_block cta) {
  unsigned int offset = 1;
  const unsigned int tid = threadIdx.x;

  // higher levels of scan tree
  for (int d = (num_threads_compaction >> 1); d > 0; d >>= 1) {
    cg::sync(cta);

    if (tid < d) {
      unsigned int ai = offset * (2 * tid + 1) - 1;
      unsigned int bi = offset * (2 * tid + 2) - 1;

      s_compaction_list_exc[bi] =
          s_compaction_list_exc[bi] + s_compaction_list_exc[ai];
    }

    offset <<= 1;
  }

  // traverse down tree: first down to level 2 across
  for (int d = 2; d < num_threads_compaction; d <<= 1) {
    offset >>= 1;
    cg::sync(cta);

    if (tid < (d - 1)) {
      unsigned int ai = offset * (tid + 1) - 1;
      unsigned int bi = ai + (offset >> 1);

      s_compaction_list_exc[bi] =
          s_compaction_list_exc[bi] + s_compaction_list_exc[ai];
    }
  }

  cg::sync(cta);
}

///////////////////////////////////////////////////////////////////////////////
//! Perform stream compaction for second child intervals
//! @param  s_left  shared
//! @param  s_left  shared memory storage for left interval limits
//! @param  s_right  shared memory storage for right interval limits
//! @param  s_left_count  shared memory storage for number of eigenvalues less
//!                       than left interval limits
//! @param  s_right_count  shared memory storage for number of eigenvalues less
//!                       than right interval limits
//! @param  mid    midpoint of current interval (left of new interval)
//! @param  right  upper limit of interval
//! @param  mid_count  eigenvalues less than \a mid
//! @param  s_compaction_list  list containing the indices where the data has
//!         to be stored
//! @param  num_threads_active  number of active threads / intervals
//! @is_active_interval  mark is thread has a second non-empty child interval
///////////////////////////////////////////////////////////////////////////////
template <class T>
__device__ void compactIntervals(float *s_left, float *s_right, T *s_left_count,
                                 T *s_right_count, float mid, float right,
                                 unsigned int mid_count,
                                 unsigned int right_count, T *s_compaction_list,
                                 unsigned int num_threads_active,
                                 unsigned int is_active_second) {
  const unsigned int tid = threadIdx.x;

  // perform compaction / copy data for all threads where the second
  // child is not dead
  if ((tid < num_threads_active) && (1 == is_active_second)) {
    unsigned int addr_w = num_threads_active + s_compaction_list[tid];

    s_left[addr_w] = mid;
    s_right[addr_w] = right;
    s_left_count[addr_w] = mid_count;
    s_right_count[addr_w] = right_count;
  }
}

///////////////////////////////////////////////////////////////////////////////
//! Store intervals that have already converged (w.r.t. the desired precision),
//! duplicating intervals that contain multiple eigenvalues
//! @param  s_left  shared memory storage for left interval limits
//! @param  s_right  shared memory storage for right interval limits
//! @param  s_left_count  shared memory storage for number of eigenvalues less
//!                       than left interval limits
//! @param  s_right_count  shared memory storage for number of eigenvalues less
//!                       than right interval limits
//! @param  left   lower limit of interval
//! @param  mid    midpoint of interval (updated if split is necessary)
//! @param  right  upper limit of interval
//! @param  left_count  eigenvalues less than \a left
//! @param  mid_count  eigenvalues less than \a mid
//! @param  right_count  eigenvalues less than \a right
//! @param  s_compaction_list_exc  helper array for stream compaction, updated
//!                                at tid if split is necessary
//! @param  compact_second_chunk  shared mem flag if second chunk is used and
//!                               ergo requires compaction
//! @param  num_threads_active  number of active threads / intervals
///////////////////////////////////////////////////////////////////////////////
template <class T, class S>
__device__ void storeIntervalConverged(float *s_left, float *s_right,
                                       T *s_left_count, T *s_right_count,
                                       float &left, float &mid, float &right,
                                       S &left_count, S &mid_count,
                                       S &right_count, T *s_compaction_list_exc,
                                       unsigned int &compact_second_chunk,
                                       const unsigned int num_threads_active) {
  const unsigned int tid = threadIdx.x;
  const unsigned int multiplicity = right_count - left_count;

  // check multiplicity of eigenvalue
  if (1 == multiplicity) {
    // just re-store intervals, simple eigenvalue
    s_left[tid] = left;
    s_right[tid] = right;
    s_left_count[tid] = left_count;
    s_right_count[tid] = right_count;

    // mark that no second child / clear
    s_right_count[tid + num_threads_active] = 0;
    s_compaction_list_exc[tid] = 0;
  } else {
    // number of eigenvalues after the split less than mid
    mid_count = left_count + (multiplicity >> 1);

    // store left interval
    s_left[tid] = left;
    s_right[tid] = right;
    s_left_count[tid] = left_count;
    s_right_count[tid] = mid_count;

    mid = left;

    // mark that second child interval exists
    s_right_count[tid + num_threads_active] = right_count;
    s_compaction_list_exc[tid] = 1;
    compact_second_chunk = 1;
  }
}

template <class T, class S>
__device__ void storeIntervalConverged(float *s_left, float *s_right,
                                       T *s_left_count, T *s_right_count,
                                       float &left, float &mid, float &right,
                                       S &left_count, S &mid_count,
                                       S &right_count, T *s_compaction_list_exc,
                                       unsigned int &compact_second_chunk,
                                       const unsigned int num_threads_active,
                                       unsigned int &is_active_second) {
  const unsigned int tid = threadIdx.x;
  const unsigned int multiplicity = right_count - left_count;

  // check multiplicity of eigenvalue
  if (1 == multiplicity) {
    // just re-store intervals, simple eigenvalue
    s_left[tid] = left;
    s_right[tid] = right;
    s_left_count[tid] = left_count;
    s_right_count[tid] = right_count;

    // mark that no second child / clear
    is_active_second = 0;
    s_compaction_list_exc[tid] = 0;
  } else {
    // number of eigenvalues after the split less than mid
    mid_count = left_count + (multiplicity >> 1);

    // store left interval
    s_left[tid] = left;
    s_right[tid] = right;
    s_left_count[tid] = left_count;
    s_right_count[tid] = mid_count;

    mid = left;

    // mark that second child interval exists
    is_active_second = 1;
    s_compaction_list_exc[tid] = 1;
    compact_second_chunk = 1;
  }
}

///////////////////////////////////////////////////////////////////////////////
//! Subdivide interval if active and not already converged
//! @param tid  id of thread
//! @param  s_left  shared memory storage for left interval limits
//! @param  s_right  shared memory storage for right interval limits
//! @param  s_left_count  shared memory storage for number of eigenvalues less
//!                       than left interval limits
//! @param  s_right_count  shared memory storage for number of eigenvalues less
//!                       than right interval limits
//! @param  num_threads_active  number of active threads in warp
//! @param  left   lower limit of interval
//! @param  right  upper limit of interval
//! @param  left_count  eigenvalues less than \a left
//! @param  right_count  eigenvalues less than \a right
//! @param  all_threads_converged  shared memory flag if all threads are
//!                                 converged
///////////////////////////////////////////////////////////////////////////////
template <class T>
__device__ void subdivideActiveInterval(
    const unsigned int tid, float *s_left, float *s_right, T *s_left_count,
    T *s_right_count, const unsigned int num_threads_active, float &left,
    float &right, unsigned int &left_count, unsigned int &right_count,
    float &mid, unsigned int &all_threads_converged) {
  // for all active threads
  if (tid < num_threads_active) {
    left = s_left[tid];
    right = s_right[tid];
    left_count = s_left_count[tid];
    right_count = s_right_count[tid];

    // check if thread already converged
    if (left != right) {
      mid = computeMidpoint(left, right);
      atomicExch(&all_threads_converged, 0);
    } else if ((right_count - left_count) > 1) {
      // mark as not converged if multiple eigenvalues enclosed
      // duplicate interval in storeIntervalsConverged()
      atomicExch(&all_threads_converged, 0);
    }

  }  // end for all active threads
}

#endif  // #ifndef _BISECT_UTIL_H_
