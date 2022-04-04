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

// Definitions for GPU quicksort

#ifndef QUICKSORT_H
#define QUICKSORT_H

#define QSORT_BLOCKSIZE_SHIFT 9
#define QSORT_BLOCKSIZE (1 << QSORT_BLOCKSIZE_SHIFT)
#define BITONICSORT_LEN 1024  // Must be power of 2!
#define QSORT_MAXDEPTH \
  16  // Will force final bitonic stage at depth QSORT_MAXDEPTH+1

////////////////////////////////////////////////////////////////////////////////
// The algorithm uses several variables updated by using atomic operations.
////////////////////////////////////////////////////////////////////////////////
typedef struct __align__(128) qsortAtomicData_t {
  volatile unsigned int lt_offset;     // Current output offset for <pivot
  volatile unsigned int gt_offset;     // Current output offset for >pivot
  volatile unsigned int sorted_count;  // Total count sorted, for deciding when
                                       // to launch next wave
  volatile unsigned int
      index;  // Ringbuf tracking index. Can be ignored if not using ringbuf.
}
qsortAtomicData;

////////////////////////////////////////////////////////////////////////////////
// A ring-buffer for rapid stack allocation
////////////////////////////////////////////////////////////////////////////////
typedef struct qsortRingbuf_t {
  volatile unsigned int head;  // Head pointer - we allocate from here
  volatile unsigned int
      tail;  // Tail pointer - indicates last still-in-use element
  volatile unsigned int count;  // Total count allocated
  volatile unsigned int max;    // Max index allocated
  unsigned int stacksize;    // Wrap-around size of buffer (must be power of 2)
  volatile void *stackbase;  // Pointer to the stack we're allocating from
} qsortRingbuf;

// Stack elem count must be power of 2!
#define QSORT_STACK_ELEMS \
  1 * 1024 * 1024  // One million stack elements is a HUGE number.

__global__ void qsort_warp(unsigned *indata, unsigned *outdata,
                           unsigned int len, qsortAtomicData *atomicData,
                           qsortRingbuf *ringbuf, unsigned int source_is_indata,
                           unsigned int depth);
__global__ void bitonicsort(unsigned *indata, unsigned *outdata,
                            unsigned int offset, unsigned int len);
__global__ void big_bitonicsort(unsigned *indata, unsigned *outdata,
                                unsigned *backbuf, unsigned int offset,
                                unsigned int len);

#endif  // QUICKSORT_H
