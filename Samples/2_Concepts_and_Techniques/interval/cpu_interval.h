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

/* Simple CPU implementation
*  Depends on Boost.Interval
*/

#ifndef CPU_INTERVAL_H
#define CPU_INTERVAL_H

#ifndef __USE_ISOC99
#define __USE_ISOC99
#endif

#include <iostream>
#include <vector>
#include <boost/numeric/interval.hpp>
//#include <iomanip>

#define UNPROTECTED 0
#define USE_RECURSION_CPU 1

using boost::numeric::interval;
using namespace boost::numeric;

template <class T, int N, int THREADS>
class global_stack_cpu {
 private:
  T *buf;
  int free_index;

 public:
  // buf should point to an allocated global buffer of size N * THREADS *
  // sizeof(T)
  global_stack_cpu(T *buf, int thread_id) : buf(buf), free_index(thread_id) {}

  void push(T const &v) {
    buf[free_index] = v;
    free_index += THREADS;
  }
  T pop() {
    free_index -= THREADS;
    return buf[free_index];
  }
  bool full() { return free_index >= N * THREADS; }
  bool empty() { return free_index < THREADS; }
  int size() { return free_index / THREADS; }
};

// The function F of which we want to find roots, defined on intervals
// Should typically depend on thread_id (indexing an array of coefficients...)
template <class I>
I f_cpu(I const &x, int thread_id) {
  typedef typename I::base_type T;
  T alpha = -T(thread_id) / T(THREADS);
  return square(x - I(1)) + I(alpha) * x;
}

// First derivative of F, also defined on intervals
template <class I>
I fd_cpu(I const &x, int thread_id) {
  typedef typename I::base_type T;
  T alpha = -T(thread_id) / T(THREADS);
  return I(2) * x + I(alpha - 2);
}

// Is this interval small enough to stop iterating?
template <class I>
bool is_minimal_cpu(I const &x, int thread_id) {
  typedef typename I::base_type T;
  T const epsilon_x = 1e-6f;
  T const epsilon_y = 1e-6f;
  return !empty(x) && (width(x) <= epsilon_x * abs(median(x)) ||
                       width(f_cpu(x, thread_id)) <= epsilon_y);
}

// In some cases, Newton iterations converge slowly.
// Bisecting the interval accelerates convergence.
template <class I>
bool should_bisect_cpu(I const &x, I const &x1, I const &x2,
                       typename I::base_type alpha) {
  typedef typename I::base_type T;
  T wmax = alpha * width(x);
  return width(x1) > wmax || width(x2) > wmax;
}

int const DEPTH_WORK = 128;

// Main interval Newton loop.
// Keep refining a list of intervals stored in a stack.
// Always keep the next interval to work on in registers (avoids excessive
// spilling to local mem)
template <class I, int THREADS, int DEPTH_RESULT>
void newton_interval_cpu(global_stack_cpu<I, DEPTH_RESULT, THREADS> &result,
                         I const &ix0, int thread_id) {
  typedef typename I::base_type T;

  T const alpha = .99f;  // Threshold before switching to bisection

  // Intervals to be processed
  I local_buffer[DEPTH_WORK];
  global_stack_cpu<I, DEPTH_WORK, 1> work(local_buffer, 0);

  // We start with the whole domain
  I ix = ix0;

  while (true) {
    // Compute (x - F({x})/F'(ix)) inter ix
    // -> may yield 0, 1 or 2 intervals
    T x = median(ix);
    I iq = f_cpu(I(x), thread_id);
    I id = fd_cpu(ix, thread_id);

    bool has_part2;
    I part1, part2;
    part1 = division_part1(iq, id, has_part2);
    part1 = intersect(I(x) - part1, ix);

    if (has_part2) {
      part2 = division_part2(iq, id);
      part2 = intersect(I(x) - part2, ix);
    }

    // Do we have small-enough intervals?
    if (is_minimal_cpu(part1, thread_id)) {
      result.push(part1);
      part1 = I::empty();
    }

    if (has_part2 && is_minimal_cpu(part2, thread_id)) {
      result.push(part2);
      part2 = I::empty();
    }

    if (should_bisect_cpu(ix, part1, part2, alpha)) {
      // Not so good improvement
      // Switch to bisection method for this step
      part1 = I(ix.lower(), x);
      part2 = I(x, ix.upper());
      has_part2 = true;
    }

    if ((part1.lower() <= part1.upper()) && !empty(part1)) {
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

template <class I, int THREADS, int DEPTH_RESULT>
void newton_interval_rec_cpu(global_stack_cpu<I, DEPTH_RESULT, THREADS> &result,
                             I const &ix, int thread_id) {
  typedef typename I::base_type T;
  T const alpha = .99f;  // Threshold before switching to bisection

  if (is_minimal_cpu(ix, thread_id)) {
    result.push(ix);
    return;
  }

  // Compute (x - F({x})/F'(ix)) inter ix
  // -> may yield 0, 1 or 2 intervals
  T x = median(ix);
  I iq = f_cpu(I(x), thread_id);
  I id = fd_cpu(ix, thread_id);

  bool has_part2;
  I part1, part2;
  part1 = division_part1(iq, id, has_part2);
  part1 = intersect(I(x) - part1, ix);

  if (has_part2) {
    part2 = division_part2(iq, id);
    part2 = intersect(I(x) - part2, ix);
  }

  if (should_bisect_cpu(ix, part1, part2, alpha)) {
    // Not so good improvement
    // Switch to bisection method for this step
    part1 = I(ix.lower(), x);
    part2 = I(x, ix.upper());
    has_part2 = true;
  }

  if ((part1.lower() <= part1.upper()) && (!empty(part1))) {
    newton_interval_rec_cpu<I, THREADS, DEPTH_RESULT>(result, part1, thread_id);
  }

  if (has_part2 && !empty(part2)) {
    newton_interval_rec_cpu<I, THREADS, DEPTH_RESULT>(result, part2, thread_id);
  }
}

template <class I>
void test_interval_newton_cpu(I *buffer, int *nresults, I i) {
  typedef typename I::base_type T;

  // Intervals to return
  // std::vector<I> local_buffer(BLOCK_SIZE * GRID_SIZE * DEPTH_WORK);
  for (int thread_id = 0; thread_id != BLOCK_SIZE * GRID_SIZE; ++thread_id) {
    global_stack_cpu<I, DEPTH_RESULT, THREADS> result(buffer, thread_id);

#if USE_RECURSION_CPU
    newton_interval_rec_cpu<I, THREADS>(result, i, thread_id);
#else
    newton_interval_cpu<I, THREADS>(result, i, thread_id);
#endif
    nresults[thread_id] = result.size();
  }
}

typedef interval<T, interval_lib::policies<interval_lib::rounded_math<T>,
                                           interval_lib::checking_base<T> > >
    Ibase;

#if UNPROTECTED
typedef interval_lib::unprotect<Ibase>::type I_CPU;
Ibase::traits_type::rounding rnd;
#else
typedef Ibase I_CPU;
#endif

bool checkAgainstHost(int *h_nresults, int *h_nresults_cpu, I_CPU *h_result,
                      I_CPU *h_result_cpu) {
  std::cout << "\nCheck against Host computation...\n\n";
  int success = 1;
  int success1 = 1;
  int success2 = 1;

  if (h_nresults_cpu[0] == h_nresults[0]) {
    for (int i = 0; i != h_nresults[0]; ++i) {
      TYPE diff1 = abs(h_result[THREADS * i + 0].lower() -
                       h_result_cpu[THREADS * i + 0].lower());
      TYPE diff2 = abs(h_result[THREADS * i + 0].upper() -
                       h_result_cpu[THREADS * i + 0].upper());

      if ((diff1 > 1.0e-6f) || (diff2 > 1.0e-6f)) {
        success1 = 0;
        break;
      }
    }

    // in case the two intervals are reversed
    for (int i = 0; i != h_nresults[0]; ++i) {
      TYPE diff1 =
          abs(h_result[THREADS * i + 0].lower() -
              h_result_cpu[THREADS * (h_nresults[0] - i - 1) + 0].lower());
      TYPE diff2 =
          abs(h_result[THREADS * i + 0].upper() -
              h_result_cpu[THREADS * (h_nresults[0] - i - 1) + 0].upper());

      if ((diff1 > 1.0e-6f) || (diff2 > 1.0e-6f)) {
        success2 = 0;
        break;
      }
    }

    success = success1 || success2;
  } else
    success = 0;

  return (bool)success;
}

#endif
