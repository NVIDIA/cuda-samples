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

// This is a basic, recursive bitonic sort taken from
// http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm
//
// The parallel code is based on:
// http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
//
// The multithread code is from me.

#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include "cdpQuicksort.h"

// Inline PTX call to return index of highest non-zero bit in a word
static __device__ __forceinline__ unsigned int __btflo(unsigned int word) {
  unsigned int ret;
  asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
//
//  qcompare
//
//  Comparison function. Note difference from libc standard in
//  that we take by reference, not by pointer. I can't figure
//  out how to get a template-to-pointer specialisation working.
//  Perhaps it requires a class?
//
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ int qcompare(unsigned &val1, unsigned &val2) {
  return (val1 > val2) ? 1 : (val1 == val2) ? 0 : -1;
}

////////////////////////////////////////////////////////////////////////////////
//
//  Basic any-N bitonic sort. We sort "len" elements of "indata", starting
//  from the "offset" elements into the input data array. Note that "outdata"
//  can safely be the same as "indata" for an in-place sort (we stage through
//  shared memory).
//
//  We handle non-power-of-2 sizes by padding out to the next largest power of
//  2.
//  This is the fully-generic version, for sorting arbitrary data which does not
//  have a clear "maximum" value. We track "invalid" entries in a separate array
//  to make sure that they always sorts as "max value" elements. A template
//  parameter "OOR" allows specialisation to optimise away the invalid tracking.
//
//  We can do a more specialised version for int/longlong/flat/double, in which
//  we pad out the array with max-value-of-type elements. That's another
//  function.
//
//  The last step copies from indata -> outdata... the rest is done in-place.
//  We use shared memory as temporary storage, which puts an upper limit on
//  how much data we can sort per block.
//
////////////////////////////////////////////////////////////////////////////////
static __device__ __forceinline__ void bitonicsort_kernel(
    unsigned *indata, unsigned *outdata, unsigned int offset, unsigned int len,
    cg::thread_block cta) {
  __shared__ unsigned
      sortbuf[1024];  // Max of 1024 elements - TODO: make this dynamic

  // First copy data into shared memory.
  unsigned int inside = (threadIdx.x < len);
  sortbuf[threadIdx.x] = inside ? indata[threadIdx.x + offset] : 0xffffffffu;
  cg::sync(cta);

  // Now the sort loops
  // Here, "k" is the sort level (remember bitonic does a multi-level butterfly
  // style sort)
  // and "j" is the partner element in the butterfly.
  // Two threads each work on one butterfly, because the read/write needs to
  // happen
  // simultaneously
  for (unsigned int k = 2; k <= blockDim.x;
       k *= 2)  // Butterfly stride increments in powers of 2
  {
    for (unsigned int j = k >> 1; j > 0;
         j >>= 1)  // Strides also in powers of to, up to <k
    {
      unsigned int swap_idx =
          threadIdx.x ^ j;  // Index of element we're compare-and-swapping with
      unsigned my_elem = sortbuf[threadIdx.x];
      unsigned swap_elem = sortbuf[swap_idx];

      cg::sync(cta);

      // The k'th bit of my threadid (and hence my sort item ID)
      // determines if we sort ascending or descending.
      // However, since threads are reading from the top AND the bottom of
      // the butterfly, if my ID is > swap_idx, then ascending means mine<swap.
      // Finally, if either my_elem or swap_elem is out of range, then it
      // ALWAYS acts like it's the largest number.
      // Confusing? It saves us two writes though.
      unsigned int ascend = k * (swap_idx < threadIdx.x);
      unsigned int descend = k * (swap_idx > threadIdx.x);
      bool swap = false;

      if ((threadIdx.x & k) == ascend) {
        if (my_elem > swap_elem) swap = true;
      }

      if ((threadIdx.x & k) == descend) {
        if (my_elem < swap_elem) swap = true;
      }

      // If we had to swap, then write my data to the other element's position.
      // Don't forget to track out-of-range status too!
      if (swap) {
        sortbuf[swap_idx] = my_elem;
      }

      cg::sync(cta);
    }
  }

  // Copy the sorted data from shared memory back to the output buffer
  if (threadIdx.x < len) outdata[threadIdx.x + offset] = sortbuf[threadIdx.x];
}

//////////////////////////////////////////////////////////////////////////////////
//  This is an emergency-CTA sort, which sorts an arbitrary sized chunk
//  using a single block. Useful for if qsort runs out of nesting depth.
//
//  Note that bitonic sort needs enough storage to pad up to the nearest
//  power of 2. This means that the double-buffer is always large enough
//  (when combined with the main buffer), but we do not get enough space
//  to keep OOR information.
//
//  This in turn means that this sort does not work with a generic data
//  type. It must be a directly-comparable (i.e. with max value) type.
//
////////////////////////////////////////////////////////////////////////////////
static __device__ __forceinline__ void big_bitonicsort_kernel(
    unsigned *indata, unsigned *outdata, unsigned *backbuf, unsigned int offset,
    unsigned int len, cg::thread_block cta) {
  unsigned int len2 =
      1 << (__btflo(len - 1U) + 1);  // Round up len to nearest power-of-2

  if (threadIdx.x >= len2)
    return;  // Early out for case where more threads launched than there is
             // data

  // First, set up our unused values to be the max data type.
  for (unsigned int i = len; i < len2; i += blockDim.x) {
    unsigned int index = i + threadIdx.x;

    if (index < len2) {
      // Must split our index between two buffers
      if (index < len)
        indata[index + offset] = 0xffffffffu;
      else
        backbuf[index + offset - len] = 0xffffffffu;
    }
  }

  cg::sync(cta);

  // Now the sort loops
  // Here, "k" is the sort level (remember bitonic does a multi-level butterfly
  // style sort)
  // and "j" is the partner element in the butterfly.
  // Two threads each work on one butterfly, because the read/write needs to
  // happen
  // simultaneously
  for (unsigned int k = 2; k <= len2;
       k *= 2)  // Butterfly stride increments in powers of 2
  {
    for (unsigned int j = k >> 1; j > 0;
         j >>= 1)  // Strides also in powers of to, up to <k
    {
      for (unsigned int i = 0; i < len2; i += blockDim.x) {
        unsigned int index = threadIdx.x + i;
        unsigned int swap_idx =
            index ^ j;  // Index of element we're compare-and-swapping with

        // Only do the swap for index<swap_idx (avoids collision between other
        // threads)
        if (swap_idx > index) {
          unsigned my_elem, swap_elem;

          if (index < len)
            my_elem = indata[index + offset];
          else
            my_elem = backbuf[index + offset - len];

          if (swap_idx < len)
            swap_elem = indata[swap_idx + offset];
          else
            swap_elem = backbuf[swap_idx + offset - len];

          // The k'th bit of my index (and hence my sort item ID)
          // determines if we sort ascending or descending.
          // Also, if either my_elem or swap_elem is out of range, then it
          // ALWAYS acts like it's the largest number.
          bool swap = false;

          if ((index & k) == 0) {
            if (my_elem > swap_elem) swap = true;
          }

          if ((index & k) == k) {
            if (my_elem < swap_elem) swap = true;
          }

          // If we had to swap, then write my data to the other element's
          // position.
          if (swap) {
            if (swap_idx < len)
              indata[swap_idx + offset] = my_elem;
            else
              backbuf[swap_idx + offset - len] = my_elem;

            if (index < len)
              indata[index + offset] = swap_elem;
            else
              backbuf[index + offset - len] = swap_elem;
          }
        }
      }

      cg::sync(cta);  // Only need to sync for each "j" pass
    }
  }

  // Copy the sorted data from the input to the output buffer, because we sort
  // in-place
  if (outdata != indata) {
    for (unsigned int i = 0; i < len; i += blockDim.x) {
      unsigned int index = i + threadIdx.x;

      if (index < len) outdata[index + offset] = indata[index + offset];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// KERNELS
////////////////////////////////////////////////////////////////////////////////

__global__ void bitonicsort(unsigned *indata, unsigned *outdata,
                            unsigned int offset, unsigned int len) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  bitonicsort_kernel(indata, outdata, offset, len, cta);
}

__global__ void big_bitonicsort(unsigned *indata, unsigned *outdata,
                                unsigned *backbuf, unsigned int offset,
                                unsigned int len) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  big_bitonicsort_kernel(indata, outdata, backbuf, offset, len, cta);
}

////////////////////////////////////////////////////////////////////////////////
