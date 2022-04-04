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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sortingNetworks_common.h"

////////////////////////////////////////////////////////////////////////////////
// Validate sorted keys array (check for integrity and proper order)
////////////////////////////////////////////////////////////////////////////////
extern "C" uint validateSortedKeys(uint *resKey, uint *srcKey, uint batchSize,
                                   uint arrayLength, uint numValues, uint dir) {
  uint *srcHist;
  uint *resHist;

  if (arrayLength < 2) {
    printf("validateSortedKeys(): arrayLength too short, exiting...\n");
    return 1;
  }

  printf("...inspecting keys array: ");

  srcHist = (uint *)malloc(numValues * sizeof(uint));
  resHist = (uint *)malloc(numValues * sizeof(uint));

  int flag = 1;

  for (uint j = 0; j < batchSize;
       j++, srcKey += arrayLength, resKey += arrayLength) {
    // Build histograms for keys arrays
    memset(srcHist, 0, numValues * sizeof(uint));
    memset(resHist, 0, numValues * sizeof(uint));

    for (uint i = 0; i < arrayLength; i++) {
      if (srcKey[i] < numValues && resKey[i] < numValues) {
        srcHist[srcKey[i]]++;
        resHist[resKey[i]]++;
      } else {
        flag = 0;
        break;
      }
    }

    if (!flag) {
      printf("***Set %u source/result key arrays are not limited properly***\n",
             j);
      goto brk;
    }

    // Compare the histograms
    for (uint i = 0; i < numValues; i++)
      if (srcHist[i] != resHist[i]) {
        flag = 0;
        break;
      }

    if (!flag) {
      printf("***Set %u source/result keys histograms do not match***\n", j);
      goto brk;
    }

    if (dir) {
      // Ascending order
      for (uint i = 0; i < arrayLength - 1; i++)
        if (resKey[i + 1] < resKey[i]) {
          flag = 0;
          break;
        }
    } else {
      // Descending order
      for (uint i = 0; i < arrayLength - 1; i++)
        if (resKey[i + 1] > resKey[i]) {
          flag = 0;
          break;
        }
    }

    if (!flag) {
      printf("***Set %u result key array is not ordered properly***\n", j);
      goto brk;
    }
  }

brk:
  free(resHist);
  free(srcHist);

  if (flag) printf("OK\n");

  return flag;
}

extern "C" int validateValues(uint *resKey, uint *resVal, uint *srcKey,
                              uint batchSize, uint arrayLength) {
  int correctFlag = 1, stableFlag = 1;

  printf("...inspecting keys and values array: ");

  for (uint i = 0; i < batchSize;
       i++, resKey += arrayLength, resVal += arrayLength) {
    for (uint j = 0; j < arrayLength; j++) {
      if (resKey[j] != srcKey[resVal[j]]) correctFlag = 0;

      if ((j < arrayLength - 1) && (resKey[j] == resKey[j + 1]) &&
          (resVal[j] > resVal[j + 1]))
        stableFlag = 0;
    }
  }

  printf(correctFlag ? "OK\n" : "***corrupted!!!***\n");
  printf(stableFlag ? "...stability property: stable!\n"
                    : "...stability property: NOT stable\n");

  return correctFlag;
}
