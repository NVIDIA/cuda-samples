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

////////////////////////////////////////////////////////////////////////////////
//
//  cdpAdvancedQuicksort.cu
//
//  Implementation of a parallel quicksort in CUDA. It comes in
//  several parts:
//
//  1. A small-set insertion sort. We do this on any set with <=32 elements
//  2. A partitioning kernel, which - given a pivot - separates an input
//     array into elements <=pivot, and >pivot. Two quicksorts will then
//     be launched to resolve each of these.
//  3. A quicksort co-ordinator, which figures out what kernels to launch
//     and when.
//
////////////////////////////////////////////////////////////////////////////////
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <helper_cuda.h>
#include <helper_string.h>
#include "cdpQuicksort.h"

////////////////////////////////////////////////////////////////////////////////
// Inline PTX call to return index of highest non-zero bit in a word
////////////////////////////////////////////////////////////////////////////////
static __device__ __forceinline__ unsigned int __qsflo(unsigned int word) {
  unsigned int ret;
  asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
//
//  ringbufAlloc
//
//  Allocates from a ringbuffer. Allows for not failing when we run out
//  of stack for tracking the offset counts for each sort subsection.
//
//  We use the atomicMax trick to allow out-of-order retirement. If we
//  hit the size limit on the ringbuffer, then we spin-wait for people
//  to complete.
//
////////////////////////////////////////////////////////////////////////////////
template <typename T>
static __device__ T *ringbufAlloc(qsortRingbuf *ringbuf) {
  // Wait for there to be space in the ring buffer. We'll retry only a fixed
  // number of times and then fail, to avoid an out-of-memory deadlock.
  unsigned int loop = 10000;

  while (((ringbuf->head - ringbuf->tail) >= ringbuf->stacksize) &&
         (loop-- > 0))
    ;

  if (loop == 0) return NULL;

  // Note that the element includes a little index book-keeping, for freeing
  // later.
  unsigned int index = atomicAdd((unsigned int *)&ringbuf->head, 1);
  T *ret = (T *)(ringbuf->stackbase) + (index & (ringbuf->stacksize - 1));
  ret->index = index;

  return ret;
}

////////////////////////////////////////////////////////////////////////////////
//
//  ringBufFree
//
//  Releases an element from the ring buffer. If every element is released
//  up to and including this one, we can advance the tail to indicate that
//  space is now available.
//
////////////////////////////////////////////////////////////////////////////////
template <typename T>
static __device__ void ringbufFree(qsortRingbuf *ringbuf, T *data) {
  unsigned int index = data->index;  // Non-wrapped index to free
  unsigned int count = atomicAdd((unsigned int *)&(ringbuf->count), 1) + 1;
  unsigned int max = atomicMax((unsigned int *)&(ringbuf->max), index + 1);

  // Update the tail if need be. Note we update "max" to be the new value in
  // ringbuf->max
  if (max < (index + 1)) max = index + 1;

  if (max == count) atomicMax((unsigned int *)&(ringbuf->tail), count);
}

////////////////////////////////////////////////////////////////////////////////
//
//  qsort_warp
//
//  Simplest possible implementation, does a per-warp quicksort with no
//  inter-warp
//  communication. This has a high atomic issue rate, but the rest should
//  actually
//  be fairly quick because of low work per thread.
//
//  A warp finds its section of the data, then writes all data <pivot to one
//  buffer and all data >pivot to the other. Atomics are used to get a unique
//  section of the buffer.
//
//  Obvious optimisation: do multiple chunks per warp, to increase in-flight
//  loads
//  and cover the instruction overhead.
//
////////////////////////////////////////////////////////////////////////////////
__global__ void qsort_warp(unsigned *indata, unsigned *outdata,
                           unsigned int offset, unsigned int len,
                           qsortAtomicData *atomicData,
                           qsortRingbuf *atomicDataStack,
                           unsigned int source_is_indata, unsigned int depth) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Find my data offset, based on warp ID
  unsigned int thread_id = threadIdx.x + (blockIdx.x << QSORT_BLOCKSIZE_SHIFT);
  // unsigned int warp_id = threadIdx.x >> 5;   // Used for debug only
  unsigned int lane_id = threadIdx.x & (warpSize - 1);

  // Exit if I'm outside the range of sort to be done
  if (thread_id >= len) return;

  //
  // First part of the algorithm. Each warp counts the number of elements that
  // are
  // greater/less than the pivot.
  //
  // When a warp knows its count, it updates an atomic counter.
  //

  // Read in the data and the pivot. Arbitrary pivot selection for now.
  unsigned pivot = indata[offset + len / 2];
  unsigned data = indata[offset + thread_id];

  // Count how many are <= and how many are > pivot.
  // If all are <= pivot then we adjust the comparison
  // because otherwise the sort will move nothing and
  // we'll iterate forever.
  cg::coalesced_group active = cg::coalesced_threads();
  unsigned int greater = (data > pivot);
  unsigned int gt_mask = active.ballot(greater);

  if (gt_mask == 0) {
    greater = (data >= pivot);
    gt_mask = active.ballot(greater);  // Must re-ballot for adjusted comparator
  }

  unsigned int lt_mask = active.ballot(!greater);
  unsigned int gt_count = __popc(gt_mask);
  unsigned int lt_count = __popc(lt_mask);

  // Atomically adjust the lt_ and gt_offsets by this amount. Only one thread
  // need do this. Share the result using shfl
  unsigned int lt_offset, gt_offset;

  if (lane_id == 0) {
    if (lt_count > 0)
      lt_offset = atomicAdd((unsigned int *)&atomicData->lt_offset, lt_count);

    if (gt_count > 0)
      gt_offset =
          len - (atomicAdd((unsigned int *)&atomicData->gt_offset, gt_count) +
                 gt_count);
  }

  lt_offset =
      active.shfl((int)lt_offset, 0);  // Everyone pulls the offsets from lane 0
  gt_offset = active.shfl((int)gt_offset, 0);

  // Now compute my own personal offset within this. I need to know how many
  // threads with a lane ID less than mine are going to write to the same buffer
  // as me. We can use popc to implement a single-operation warp scan in this
  // case.
  unsigned lane_mask_lt;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
  unsigned int my_mask = greater ? gt_mask : lt_mask;
  unsigned int my_offset = __popc(my_mask & lane_mask_lt);

  // Move data.
  my_offset += greater ? gt_offset : lt_offset;
  outdata[offset + my_offset] = data;

  // Count up if we're the last warp in. If so, then Kepler will launch the next
  // set of sorts directly from here.
  if (lane_id == 0) {
    // Count "elements written". If I wrote the last one, then trigger the next
    // qsorts
    unsigned int mycount = lt_count + gt_count;

    if (atomicAdd((unsigned int *)&atomicData->sorted_count, mycount) +
            mycount ==
        len) {
      // We're the last warp to do any sorting. Therefore it's up to us to
      // launch the next stage.
      unsigned int lt_len = atomicData->lt_offset;
      unsigned int gt_len = atomicData->gt_offset;

      cudaStream_t lstream, rstream;
      cudaStreamCreateWithFlags(&lstream, cudaStreamNonBlocking);
      cudaStreamCreateWithFlags(&rstream, cudaStreamNonBlocking);

      // Begin by freeing our atomicData storage. It's better for the ringbuffer
      // algorithm
      // if we free when we're done, rather than re-using (makes for less
      // fragmentation).
      ringbufFree<qsortAtomicData>(atomicDataStack, atomicData);

      // Exceptional case: if "lt_len" is zero, then all values in the batch
      // are equal. We are then done (may need to copy into correct buffer,
      // though)
      if (lt_len == 0) {
        if (source_is_indata)
          cudaMemcpyAsync(indata + offset, outdata + offset,
                          gt_len * sizeof(unsigned), cudaMemcpyDeviceToDevice,
                          lstream);

        return;
      }

      // Start with lower half first
      if (lt_len > BITONICSORT_LEN) {
        // If we've exceeded maximum depth, fall through to backup
        // big_bitonicsort
        if (depth >= QSORT_MAXDEPTH) {
          // The final bitonic stage sorts in-place in "outdata". We therefore
          // re-use "indata" as the out-of-range tracking buffer. For (2^n)+1
          // elements we need (2^(n+1)) bytes of oor buffer. The backup qsort
          // buffer is at least this large when sizeof(QTYPE) >= 2.
          big_bitonicsort<<<1, BITONICSORT_LEN, 0, lstream>>>(
              outdata, source_is_indata ? indata : outdata, indata, offset,
              lt_len);
        } else {
          // Launch another quicksort. We need to allocate more storage for the
          // atomic data.
          if ((atomicData = ringbufAlloc<qsortAtomicData>(atomicDataStack)) ==
              NULL)
            printf("Stack-allocation error. Failing left child launch.\n");
          else {
            atomicData->lt_offset = atomicData->gt_offset =
                atomicData->sorted_count = 0;
            unsigned int numblocks =
                (unsigned int)(lt_len + (QSORT_BLOCKSIZE - 1)) >>
                QSORT_BLOCKSIZE_SHIFT;
            qsort_warp<<<numblocks, QSORT_BLOCKSIZE, 0, lstream>>>(
                outdata, indata, offset, lt_len, atomicData, atomicDataStack,
                !source_is_indata, depth + 1);
          }
        }
      } else if (lt_len > 1) {
        // Final stage uses a bitonic sort instead. It's important to
        // make sure the final stage ends up in the correct (original) buffer.
        // We launch the smallest power-of-2 number of threads that we can.
        unsigned int bitonic_len = 1 << (__qsflo(lt_len - 1U) + 1);
        bitonicsort<<<1, bitonic_len, 0, lstream>>>(
            outdata, source_is_indata ? indata : outdata, offset, lt_len);
      }
      // Finally, if we sorted just one single element, we must still make
      // sure that it winds up in the correct place.
      else if (source_is_indata && (lt_len == 1))
        indata[offset] = outdata[offset];

      if (cudaPeekAtLastError() != cudaSuccess)
        printf("Left-side launch fail: %s\n",
               cudaGetErrorString(cudaGetLastError()));

      // Now the upper half.
      if (gt_len > BITONICSORT_LEN) {
        // If we've exceeded maximum depth, fall through to backup
        // big_bitonicsort
        if (depth >= QSORT_MAXDEPTH)
          big_bitonicsort<<<1, BITONICSORT_LEN, 0, rstream>>>(
              outdata, source_is_indata ? indata : outdata, indata,
              offset + lt_len, gt_len);
        else {
          // Allocate new atomic storage for this launch
          if ((atomicData = ringbufAlloc<qsortAtomicData>(atomicDataStack)) ==
              NULL)
            printf("Stack allocation error! Failing right-side launch.\n");
          else {
            atomicData->lt_offset = atomicData->gt_offset =
                atomicData->sorted_count = 0;
            unsigned int numblocks =
                (unsigned int)(gt_len + (QSORT_BLOCKSIZE - 1)) >>
                QSORT_BLOCKSIZE_SHIFT;
            qsort_warp<<<numblocks, QSORT_BLOCKSIZE, 0, rstream>>>(
                outdata, indata, offset + lt_len, gt_len, atomicData,
                atomicDataStack, !source_is_indata, depth + 1);
          }
        }
      } else if (gt_len > 1) {
        unsigned int bitonic_len = 1 << (__qsflo(gt_len - 1U) + 1);
        bitonicsort<<<1, bitonic_len, 0, rstream>>>(
            outdata, source_is_indata ? indata : outdata, offset + lt_len,
            gt_len);
      } else if (source_is_indata && (gt_len == 1))
        indata[offset + lt_len] = outdata[offset + lt_len];

      if (cudaPeekAtLastError() != cudaSuccess)
        printf("Right-side launch fail: %s\n",
               cudaGetErrorString(cudaGetLastError()));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//
//  run_quicksort
//
//  Host-side code to run the Kepler version of quicksort. It's pretty
//  simple, because all launch control is handled on the device via CDP.
//
//  All parallel quicksorts require an equal-sized scratch buffer. This
//  must be passed in ahead of time.
//
//  Returns the time elapsed for the sort.
//
////////////////////////////////////////////////////////////////////////////////
float run_quicksort_cdp(unsigned *gpudata, unsigned *scratchdata,
                        unsigned int count, cudaStream_t stream) {
  unsigned int stacksize = QSORT_STACK_ELEMS;

  // This is the stack, for atomic tracking of each sort's status
  qsortAtomicData *gpustack;
  checkCudaErrors(
      cudaMalloc((void **)&gpustack, stacksize * sizeof(qsortAtomicData)));
  checkCudaErrors(cudaMemset(
      gpustack, 0, sizeof(qsortAtomicData)));  // Only need set first entry to 0

  // Create the memory ringbuffer used for handling the stack.
  // Initialise everything to where it needs to be.
  qsortRingbuf buf;
  qsortRingbuf *ringbuf;
  checkCudaErrors(cudaMalloc((void **)&ringbuf, sizeof(qsortRingbuf)));
  buf.head = 1;  // We start with one allocation
  buf.tail = 0;
  buf.count = 0;
  buf.max = 0;
  buf.stacksize = stacksize;
  buf.stackbase = gpustack;
  checkCudaErrors(
      cudaMemcpy(ringbuf, &buf, sizeof(buf), cudaMemcpyHostToDevice));

  // Timing events...
  cudaEvent_t ev1, ev2;
  checkCudaErrors(cudaEventCreate(&ev1));
  checkCudaErrors(cudaEventCreate(&ev2));
  checkCudaErrors(cudaEventRecord(ev1));

  // Now we trivially launch the qsort kernel
  if (count > BITONICSORT_LEN) {
    unsigned int numblocks =
        (unsigned int)(count + (QSORT_BLOCKSIZE - 1)) >> QSORT_BLOCKSIZE_SHIFT;
    qsort_warp<<<numblocks, QSORT_BLOCKSIZE, 0, stream>>>(
        gpudata, scratchdata, 0U, count, gpustack, ringbuf, true, 0);
  } else {
    bitonicsort<<<1, BITONICSORT_LEN>>>(gpudata, gpudata, 0, count);
  }

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaEventRecord(ev2));
  checkCudaErrors(cudaDeviceSynchronize());

  float elapse = 0.0f;

  if (cudaPeekAtLastError() != cudaSuccess)
    printf("Launch failure: %s\n", cudaGetErrorString(cudaGetLastError()));
  else
    checkCudaErrors(cudaEventElapsedTime(&elapse, ev1, ev2));

  // Sanity check that the stack allocator is doing the right thing
  checkCudaErrors(
      cudaMemcpy(&buf, ringbuf, sizeof(*ringbuf), cudaMemcpyDeviceToHost));

  if (count > BITONICSORT_LEN && buf.head != buf.tail) {
    printf("Stack allocation error!\nRingbuf:\n");
    printf("\t head = %u\n", buf.head);
    printf("\t tail = %u\n", buf.tail);
    printf("\tcount = %u\n", buf.count);
    printf("\t  max = %u\n", buf.max);
  }

  // Release our stack data once we're done
  checkCudaErrors(cudaFree(ringbuf));
  checkCudaErrors(cudaFree(gpustack));

  return elapse;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int run_qsort(unsigned int size, int seed, int debug, int loop, int verbose) {
  if (seed > 0) srand(seed);

  // Create and set up our test
  unsigned *gpudata, *scratchdata;
  checkCudaErrors(cudaMalloc((void **)&gpudata, size * sizeof(unsigned)));
  checkCudaErrors(cudaMalloc((void **)&scratchdata, size * sizeof(unsigned)));

  // Create CPU data.
  unsigned *data = new unsigned[size];
  unsigned int min = loop ? loop : size;
  unsigned int max = size;
  loop = (loop == 0) ? 1 : loop;

  for (size = min; size <= max; size += loop) {
    if (verbose) printf(" Input: ");

    for (unsigned int i = 0; i < size; i++) {
      // Build data 8 bits at a time
      data[i] = 0;
      char *ptr = (char *)&(data[i]);

      for (unsigned j = 0; j < sizeof(unsigned); j++) {
        // Easy-to-read data in debug mode
        if (debug) {
          *ptr++ = (char)(rand() % 10);
          break;
        }

        *ptr++ = (char)(rand() & 255);
      }

      if (verbose) {
        if (i && !(i % 32)) printf("\n        ");

        printf("%u ", data[i]);
      }
    }

    if (verbose) printf("\n");

    checkCudaErrors(cudaMemcpy(gpudata, data, size * sizeof(unsigned),
                               cudaMemcpyHostToDevice));

    // So we're now populated and ready to go! We size our launch as
    // blocks of up to BLOCKSIZE threads, and appropriate grid size.
    // One thread is launched per element.
    float elapse;
    elapse = run_quicksort_cdp(gpudata, scratchdata, size, NULL);

    // run_bitonicsort<SORTTYPE>(gpudata, scratchdata, size, verbose);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy back the data and verify correct sort
    checkCudaErrors(cudaMemcpy(data, gpudata, size * sizeof(unsigned),
                               cudaMemcpyDeviceToHost));

    if (verbose) {
      printf("Output: ");

      for (unsigned int i = 0; i < size; i++) {
        if (i && !(i % 32)) printf("\n        ");

        printf("%u ", data[i]);
      }

      printf("\n");
    }

    unsigned int check;

    for (check = 1; check < size; check++) {
      if (data[check] < data[check - 1]) {
        printf("FAILED at element: %d\n", check);
        break;
      }
    }

    if (check != size) {
      printf("    cdpAdvancedQuicksort FAILED\n");
      exit(EXIT_FAILURE);
    } else
      printf("    cdpAdvancedQuicksort PASSED\n");

    // Display the time between event recordings
    printf("Sorted %u elems in %.3f ms (%.3f Melems/sec)\n", size, elapse,
           (float)size / (elapse * 1000.0f));
    fflush(stdout);
  }

  // Release everything and we're done
  checkCudaErrors(cudaFree(scratchdata));
  checkCudaErrors(cudaFree(gpudata));
  delete (data);
  return 0;
}

static void usage() {
  printf(
      "Syntax: cdpAdvancedQuicksort [-size=<num>] [-seed=<num>] [-debug] "
      "[-loop-step=<num>] [-verbose]\n");
  printf(
      "If loop_step is non-zero, will run from 1->array_len in steps of "
      "loop_step\n");
}

// Host side entry
int main(int argc, char *argv[]) {
  int size = 1000000;
  unsigned int seed = 0;
  int debug = 0;
  int loop = 0;
  int verbose = 0;

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "h")) {
    usage();
    printf("&&&& cdpAdvancedQuicksort WAIVED\n");
    exit(EXIT_WAIVED);
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
    size = getCmdLineArgumentInt(argc, (const char **)argv, "size");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "seed")) {
    seed = getCmdLineArgumentInt(argc, (const char **)argv, "seed");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "loop-step")) {
    loop = getCmdLineArgumentInt(argc, (const char **)argv, "loop-step");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "debug")) {
    debug = 1;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "verbose")) {
    verbose = 1;
  }

  // Get device properties
  int cuda_device = findCudaDevice(argc, (const char **)argv);
  cudaDeviceProp properties;
  checkCudaErrors(cudaGetDeviceProperties(&properties, cuda_device));
  int cdpCapable =
      (properties.major == 3 && properties.minor >= 5) || properties.major >= 4;

  printf("GPU device %s has compute capabilities (SM %d.%d)\n", properties.name,
         properties.major, properties.minor);

  if (!cdpCapable) {
    printf(
        "cdpAdvancedQuicksort requires SM 3.5 or higher to use CUDA Dynamic "
        "Parallelism.  Exiting...\n");
    exit(EXIT_WAIVED);
  }

  printf("Running qsort on %d elements with seed %d, on %s\n", size, seed,
         properties.name);

  run_qsort(size, seed, debug, loop, verbose);

  exit(EXIT_SUCCESS);
}
