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

#ifndef SIMPLEVOTE_KERNEL_CU
#define SIMPLEVOTE_KERNEL_CU

////////////////////////////////////////////////////////////////////////////////
// Vote Any/All intrinsic kernel function tests are supported only by CUDA
// capable devices that are CUDA hardware that has SM1.2 or later
// Vote Functions (refer to section 4.4.5 in the CUDA Programming Guide)
////////////////////////////////////////////////////////////////////////////////

// Kernel #1 tests the across-the-warp vote(any) intrinsic.
// If ANY one of the threads (within the warp) of the predicated condition
// returns a non-zero value, then all threads within this warp will return a
// non-zero value
extern "C" __global__ void VoteAnyKernel1(unsigned int *input,
                                          unsigned int *result, int size) {
  int tx = threadIdx.x;

  int mask = 0xffffffff;
  result[tx] = __any_sync(mask, input[tx]);
}

// Kernel #2 tests the across-the-warp vote(all) intrinsic.
// If ALL of the threads (within the warp) of the predicated condition returns
// a non-zero value, then all threads within this warp will return a non-zero
// value
extern "C" __global__ void VoteAllKernel2(unsigned int *input,
                                          unsigned int *result, int size) {
  int tx = threadIdx.x;

  int mask = 0xffffffff;
  result[tx] = __all_sync(mask, input[tx]);
}

// Kernel #3 is a directed test for the across-the-warp vote(all) intrinsic.
// This kernel will test for conditions across warps, and within half warps
extern "C" __global__ void VoteAnyKernel3(bool *info, int warp_size) {
  int tx = threadIdx.x;
  unsigned int mask = 0xffffffff;
  bool *offs = info + (tx * 3);

  // The following should hold true for the second and third warp
  *offs = __any_sync(mask, (tx >= (warp_size * 3) / 2));
  // The following should hold true for the "upper half" of the second warp,
  // and all of the third warp
  *(offs + 1) = (tx >= (warp_size * 3) / 2 ? true : false);

  // The following should hold true for the third warp only
  if (__all_sync(mask, (tx >= (warp_size * 3) / 2))) {
    *(offs + 2) = true;
  }
}

#endif
