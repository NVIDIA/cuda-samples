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

#ifndef PERMUTATIONS_H
#define PERMUTATIONS_H

#include <helper_cuda.h>  // assert

static void computePermutations(uint permutations[1024]) {
  int indices[16];
  int num = 0;

  // 3 element permutations:

  // first cluster [0,i) is at the start
  for (int m = 0; m < 16; ++m) {
    indices[m] = 0;
  }

  const int imax = 15;

  for (int i = imax; i >= 0; --i) {
    // second cluster [i,j) is half along
    for (int m = i; m < 16; ++m) {
      indices[m] = 2;
    }

    const int jmax = (i == 0) ? 15 : 16;

    for (int j = jmax; j >= i; --j) {
      // last cluster [j,k) is at the end
      if (j < 16) {
        indices[j] = 1;
      }

      uint permutation = 0;

      for (int p = 0; p < 16; p++) {
        permutation |= indices[p] << (p * 2);
        // permutation |= indices[15-p] << (p * 2);
      }

      permutations[num] = permutation;

      num++;
    }
  }

  assert(num == 151);

  for (int i = 0; i < 9; i++) {
    permutations[num] = 0x000AA555;
    num++;
  }

  assert(num == 160);

  // Append 4 element permutations:

  // first cluster [0,i) is at the start
  for (int m = 0; m < 16; ++m) {
    indices[m] = 0;
  }

  for (int i = imax; i >= 0; --i) {
    // second cluster [i,j) is one third along
    for (int m = i; m < 16; ++m) {
      indices[m] = 2;
    }

    const int jmax = (i == 0) ? 15 : 16;

    for (int j = jmax; j >= i; --j) {
      // third cluster [j,k) is two thirds along
      for (int m = j; m < 16; ++m) {
        indices[m] = 3;
      }

      int kmax = (j == 0) ? 15 : 16;

      for (int k = kmax; k >= j; --k) {
        // last cluster [k,n) is at the end
        if (k < 16) {
          indices[k] = 1;
        }

        uint permutation = 0;

        bool hasThree = false;

        for (int p = 0; p < 16; p++) {
          permutation |= indices[p] << (p * 2);
          // permutation |= indices[15-p] << (p * 2);

          if (indices[p] == 3) hasThree = true;
        }

        if (hasThree) {
          permutations[num] = permutation;
          num++;
        }
      }
    }
  }

  assert(num == 975);

  // 1024 - 969 - 7 = 48 extra elements

  // It would be nice to set these extra elements with better values...
  for (int i = 0; i < 49; i++) {
    permutations[num] = 0x00AAFF55;
    num++;
  }

  assert(num == 1024);
}

#endif  // PERMUTATIONS_H
