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

#ifndef CUDA_INTERVAL_H
#define CUDA_INTERVAL_H

#include "interval.h"
#include "cuda_interval_lib.h"

// Stack in local memory. Managed independently for each thread.
template <class T, int N>
class local_stack {
 private:
  T buf[N];
  int tos;

 public:
  __device__ local_stack() : tos(-1) {}
  __device__ T const &top() const { return buf[tos]; }
  __device__ T &top() { return buf[tos]; }
  __device__ void push(T const &v) { buf[++tos] = v; }
  __device__ T pop() { return buf[tos--]; }
  __device__ bool full() { return tos == (N - 1); }
  __device__ bool empty() { return tos == -1; }
};

// Stacks in global memory.
// Same function as local_stack, but accessible from the host.
// Interleaved between threads by blocks of THREADS elements.
// Independent stack for each thread, no sharing of data between threads.
template <class T, int N, int THREADS>
class global_stack {
 private:
  T *buf;
  int free_index;

 public:
  // buf should point to an allocated global buffer of
  // size N * THREADS * sizeof(T)
  __device__ global_stack(T *buf, int thread_id)
      : buf(buf), free_index(thread_id) {}

  __device__ void push(T const &v) {
    buf[free_index] = v;
    free_index += THREADS;
  }
  __device__ T pop() {
    free_index -= THREADS;
    return buf[free_index];
  }
  __device__ bool full() { return free_index >= N * THREADS; }
  __device__ bool empty() { return free_index < THREADS; }
  __device__ int size() { return free_index / THREADS; }
};

// The function F of which we want to find roots, defined on intervals
// Should typically depend on thread_id (indexing an array of coefficients...)
template <class T>
__device__ interval_gpu<T> f(interval_gpu<T> const &x, int thread_id) {
  typedef interval_gpu<T> I;
  T alpha = -T(thread_id) / T(THREADS);
  return square(x - I(1)) + I(alpha) * x;
}

// First derivative of F, also defined on intervals
template <class T>
__device__ interval_gpu<T> fd(interval_gpu<T> const &x, int thread_id) {
  typedef interval_gpu<T> I;
  T alpha = -T(thread_id) / T(THREADS);
  return I(2) * x + I(alpha - 2);
}

// Is this interval small enough to stop iterating?
template <class T>
__device__ bool is_minimal(interval_gpu<T> const &x, int thread_id) {
  T const epsilon_x = 1e-6f;
  T const epsilon_y = 1e-6f;
  return !empty(x) && (width(x) <= epsilon_x * abs(median(x)) ||
                       width(f(x, thread_id)) <= epsilon_y);
}

// In some cases, Newton iterations converge slowly.
// Bisecting the interval accelerates convergence.
template <class T>
__device__ bool should_bisect(interval_gpu<T> const &x,
                              interval_gpu<T> const &x1,
                              interval_gpu<T> const &x2, T alpha) {
  T wmax = alpha * width(x);
  return (!empty(x1) && width(x1) > wmax) || (!empty(x2) && width(x2) > wmax);
}

// Main interval Newton loop.
// Keep refining a list of intervals stored in a stack.
// Always keep the next interval to work on in registers
// (avoids excessive spilling to local mem)
template <class T, int THREADS, int DEPTH_RESULT>
__device__ void newton_interval(
    global_stack<interval_gpu<T>, DEPTH_RESULT, THREADS> &result,
    interval_gpu<T> const &ix0, int thread_id) {
  typedef interval_gpu<T> I;
  int const DEPTH_WORK = 128;

  T const alpha = .99f;  // Threshold before switching to bisection

  // Intervals to be processed
  local_stack<I, DEPTH_WORK> work;

  // We start with the whole domain
  I ix = ix0;

  while (true) {
    // Compute (x - F({x})/F'(ix)) inter ix
    // -> may yield 0, 1 or 2 intervals
    T x = median(ix);
    I iq = f(I(x), thread_id);
    I id = fd(ix, thread_id);

    bool has_part2;
    I part1, part2 = I::empty();
    part1 = division_part1(iq, id, has_part2);
    part1 = intersect(I(x) - part1, ix);

    if (has_part2) {
      part2 = division_part2(iq, id);
      part2 = intersect(I(x) - part2, ix);
    }

    // Do we have small-enough intervals?
    if (is_minimal(part1, thread_id)) {
      result.push(part1);
      part1 = I::empty();
    }

    if (has_part2 && is_minimal(part2, thread_id)) {
      result.push(part2);
      part2 = I::empty();
    }

    if (should_bisect(ix, part1, part2, alpha)) {
      // Not so good improvement
      // Switch to bisection method for this step
      part1 = I(ix.lower(), x);
      part2 = I(x, ix.upper());
      has_part2 = true;
    }

    if (!empty(part1)) {
      // At least 1 solution
      // We will compute part1 next
      ix = part1;

      if (has_part2 && !empty(part2)) {
        // 2 solutions
        // Save the second solution for later
        work.push(part2);
      }
    } else if (has_part2 && !empty(part2)) {
      // 1 solution
      // Work on that next
      ix = part2;
    } else {
      // No solution
      // Do we still have work to do in the stack?
      if (work.empty())  // If not, we are done
        break;
      else
        ix = work.pop();  // Otherwise, pick an interval to work on
    }
  }
}

// Recursive implementation
template <class T, int THREADS, int DEPTH_RESULT>
__device__ void newton_interval_rec(
    global_stack<interval_gpu<T>, DEPTH_RESULT, THREADS> &result,
    interval_gpu<T> const &ix, int thread_id) {
  typedef interval_gpu<T> I;
  T const alpha = .99f;  // Threshold before switching to bisection

  if (is_minimal(ix, thread_id)) {
    result.push(ix);
    return;
  }

  // Compute (x - F({x})/F'(ix)) inter ix
  // -> may yield 0, 1 or 2 intervals
  T x = median(ix);
  I iq = f(I(x), thread_id);
  I id = fd(ix, thread_id);

  bool has_part2;
  I part1, part2 = I::empty();
  part1 = division_part1(iq, id, has_part2);
  part1 = intersect(I(x) - part1, ix);

  if (has_part2) {
    part2 = division_part2(iq, id);
    part2 = intersect(I(x) - part2, ix);
  }

  if (should_bisect(ix, part1, part2, alpha)) {
    // Not so good improvement
    // Switch to bisection method for this step
    part1 = I(ix.lower(), x);
    part2 = I(x, ix.upper());
    has_part2 = true;
  }

  if (has_part2 && !empty(part2)) {
    newton_interval_rec<T, THREADS, DEPTH_RESULT>(result, part2, thread_id);
  }

  if (!empty(part1)) {
    newton_interval_rec<T, THREADS, DEPTH_RESULT>(result, part1, thread_id);
  }
}

// Naive implementation, no attempt to keep the top of the stack in registers
template <class T, int THREADS, int DEPTH_RESULT>
__device__ void newton_interval_naive(
    global_stack<interval_gpu<T>, DEPTH_RESULT, THREADS> &result,
    interval_gpu<T> const &ix0, int thread_id) {
  typedef interval_gpu<T> I;
  int const DEPTH_WORK = 128;
  T const alpha = .99f;  // Threshold before switching to bisection

  // Intervals to be processed
  local_stack<I, DEPTH_WORK> work;

  // We start with the whole domain
  work.push(ix0);

  while (!work.empty()) {
    I ix = work.pop();

    if (is_minimal(ix, thread_id)) {
      result.push(ix);
    } else {
      // Compute (x - F({x})/F'(ix)) inter ix
      // -> may yield 0, 1 or 2 intervals
      T x = median(ix);
      I iq = f(I(x), thread_id);
      I id = fd(ix, thread_id);

      bool has_part2;
      I part1, part2 = I::empty();
      part1 = division_part1(iq, id, has_part2);
      part1 = intersect(I(x) - part1, ix);

      if (has_part2) {
        part2 = division_part2(iq, id);
        part2 = intersect(I(x) - part2, ix);
      }

      if (should_bisect(ix, part1, part2, alpha)) {
        // Not so good improvement
        // Switch to bisection method for this step
        part1 = I(ix.lower(), x);
        part2 = I(x, ix.upper());
        has_part2 = true;
      }

      if (!empty(part1)) {
        work.push(part1);
      }

      if (has_part2 && !empty(part2)) {
        work.push(part2);
      }
    }
  }
}

template <class T>
__global__ void test_interval_newton(interval_gpu<T> *buffer, int *nresults,
                                     interval_gpu<T> i,
                                     int implementation_choice) {
  int thread_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  typedef interval_gpu<T> I;

  // Intervals to return
  global_stack<I, DEPTH_RESULT, THREADS> result(buffer, thread_id);

  switch (implementation_choice) {
    case 0:
      newton_interval_naive<T, THREADS>(result, i, thread_id);
      break;

    case 1:
      newton_interval<T, THREADS>(result, i, thread_id);
      break;

#if (__CUDA_ARCH__ >= 200)

    case 2:
      newton_interval_rec<T, THREADS>(result, i, thread_id);
      break;
#endif

    default:
      newton_interval_naive<T, THREADS>(result, i, thread_id);
  }

  nresults[thread_id] = result.size();
}

#endif
