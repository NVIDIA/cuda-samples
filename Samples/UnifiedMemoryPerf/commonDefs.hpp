/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef _COMMON_DEFS_
#define _COMMON_DEFS_
#include <cuda.h>

#define ONE_KB 1024
#define ONE_MB (ONE_KB * ONE_KB)

extern size_t maxSampleSizeInMb;
extern int numKernelRuns;
extern int verboseResults;

extern unsigned int findNumSizesToTest(unsigned int minSize,
                                       unsigned int maxSize,
                                       unsigned int multiplier);

// For Tracking the different memory allocation types
typedef enum memAllocType_enum {
  MEMALLOC_TYPE_START,
  USE_MANAGED_MEMORY_WITH_HINTS = MEMALLOC_TYPE_START,
  USE_MANAGED_MEMORY_WITH_HINTS_ASYNC,
  USE_MANAGED_MEMORY,
  USE_ZERO_COPY,
  USE_HOST_PAGEABLE_AND_DEVICE_MEMORY,
  USE_HOST_PAGEABLE_AND_DEVICE_MEMORY_ASYNC,
  USE_HOST_PAGELOCKED_AND_DEVICE_MEMORY,
  USE_HOST_PAGELOCKED_AND_DEVICE_MEMORY_ASYNC,
  MEMALLOC_TYPE_END = USE_HOST_PAGELOCKED_AND_DEVICE_MEMORY_ASYNC,
  MEMALLOC_TYPE_INVALID,
  MEMALLOC_TYPE_COUNT = MEMALLOC_TYPE_INVALID
} MemAllocType;

typedef enum bandwidthType_enum {
  READ_BANDWIDTH,
  WRITE_BANDWIDTH
} BandwidthType;

extern const char *memAllocTypeStr[];
extern const char *memAllocTypeShortStr[];

struct resultsData;
struct testResults;

void createAndInitTestResults(struct testResults **results,
                              const char *testName,
                              unsigned int numMeasurements,
                              unsigned int numSizesToTest);
unsigned long *getPtrSizesToTest(struct testResults *results);

void freeTestResultsAndAllResultsData(struct testResults *results);

void createResultDataAndAddToTestResults(struct resultsData **ptrData,
                                         struct testResults *results,
                                         const char *resultsName,
                                         bool printOnlyInVerbose,
                                         bool reportAsBandwidth);
double *getPtrRunTimesInMs(struct resultsData *data, int allocType,
                           int sizeIndex);

void printResults(struct testResults *results,
                  bool print_launch_transfer_results, bool print_std_deviation);
#endif
