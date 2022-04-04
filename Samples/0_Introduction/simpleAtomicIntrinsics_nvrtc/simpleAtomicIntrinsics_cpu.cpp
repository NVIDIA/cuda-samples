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

#include <math.h>
#include <stdio.h>

#define min(a, b) (a) < (b) ? (a) : (b)
#define max(a, b) (a) > (b) ? (a) : (b)

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" int computeGold(int *gpuData, const int len);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////

int computeGold(int *gpuData, const int len) {
  int val = 0;

  for (int i = 0; i < len; ++i) {
    val += 10;
  }

  if (val != gpuData[0]) {
    printf("atomicAdd failed\n");
    return false;
  }

  val = 0;

  for (int i = 0; i < len; ++i) {
    val -= 10;
  }

  if (val != gpuData[1]) {
    printf("atomicSub failed\n");
    return false;
  }

  bool found = false;

  for (int i = 0; i < len; ++i) {
    // third element should be a member of [0, len)
    if (i == gpuData[2]) {
      found = true;
      break;
    }
  }

  if (!found) {
    printf("atomicExch failed\n");
    return false;
  }

  val = -(1 << 8);

  for (int i = 0; i < len; ++i) {
    // fourth element should be len-1
    val = max(val, i);
  }

  if (val != gpuData[3]) {
    printf("atomicMax failed\n");
    return false;
  }

  val = 1 << 8;

  for (int i = 0; i < len; ++i) {
    val = min(val, i);
  }

  if (val != gpuData[4]) {
    printf("atomicMin failed\n");
    return false;
  }

  int limit = 17;
  val = 0;

  for (int i = 0; i < len; ++i) {
    val = (val >= limit) ? 0 : val + 1;
  }

  if (val != gpuData[5]) {
    printf("atomicInc failed\n");
    return false;
  }

  limit = 137;
  val = 0;

  for (int i = 0; i < len; ++i) {
    val = ((val == 0) || (val > limit)) ? limit : val - 1;
  }

  if (val != gpuData[6]) {
    printf("atomicDec failed\n");
    return false;
  }

  found = false;

  for (int i = 0; i < len; ++i) {
    // eighth element should be a member of [0, len)
    if (i == gpuData[7]) {
      found = true;
      break;
    }
  }

  if (!found) {
    printf("atomicCAS failed\n");
    return false;
  }

  val = 0xff;
  for (int i = 0; i < len; ++i) {
    // 9th element should be 1
    val &= (2 * i + 7);
  }

  if (val != gpuData[8]) {
    printf("atomicAnd failed\n");
    return false;
  }

  val = 0;
  for (int i = 0; i < len; ++i) {
    // 10th element should be 0xff
    val |= (1 << i);
  }

  if (val != gpuData[9]) {
    printf("atomicOr failed\n");
    return false;
  }

  val = 0xff;

  for (int i = 0; i < len; ++i) {
    // 11th element should be 0xff
    val ^= i;
  }

  if (val != gpuData[10]) {
    printf("atomicXor failed\n");
    return false;
  }

  return true;
}
