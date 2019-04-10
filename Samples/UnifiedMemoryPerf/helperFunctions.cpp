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

#include <stdio.h>
#include <string.h>
#include "commonDefs.hpp"
#define CU_INIT_UUID
#include <cmath>

#define UNITS_Time "ms"
#define UNITS_BW "MB/s"
#define KB_str "KB"
#define MB_str "MB"

struct resultsData {
  char resultsName[64];
  struct testResults *results;
  // this has MEMALLOC_TYPE_COUNT * results->numSizesToTest *
  // results->numMeasurements elements
  double **runTimesInMs[MEMALLOC_TYPE_COUNT];
  double *averageRunTimesInMs[MEMALLOC_TYPE_COUNT];
  double *stdDevRunTimesInMs[MEMALLOC_TYPE_COUNT];
  double *stdDevBandwidthInMBps[MEMALLOC_TYPE_COUNT];
  bool printOnlyInVerbose;
  bool reportAsBandwidth;
  struct resultsData *next;
};

struct testResults {
  char testName[64];
  unsigned int numMeasurements;
  unsigned long *sizesToTest;
  unsigned int numSizesToTest;
  struct resultsData *resultsDataHead;
  struct resultsData *resultsDataTail;
};

unsigned int findNumSizesToTest(unsigned int minSize, unsigned int maxSize,
                                unsigned int multiplier) {
  unsigned int numSizesToTest = 0;
  while (minSize <= maxSize) {
    numSizesToTest++;
    minSize *= multiplier;
  }
  return numSizesToTest;
}

int compareDoubles(const void *ptr1, const void *ptr2) {
  return (*(double *)ptr1 > *(double *)ptr2) ? 1 : -1;
}

static inline double getTimeOrBandwidth(double runTimeInMs, unsigned long size,
                                        bool getBandwidth) {
  return (getBandwidth) ? (1000 * (size / runTimeInMs)) / ONE_MB : runTimeInMs;
}

void createAndInitTestResults(struct testResults **ptrResults,
                              const char *testName,
                              unsigned int numMeasurements,
                              unsigned int numSizesToTest) {
  unsigned int i;
  struct testResults *results;
  results = (struct testResults *)malloc(sizeof(struct testResults));
  memset(results, 0, sizeof(struct testResults));
  strcpy(results->testName, testName);
  results->numMeasurements = numMeasurements;
  results->numSizesToTest = numSizesToTest;
  results->sizesToTest =
      (unsigned long *)malloc(numSizesToTest * sizeof(unsigned long));
  results->resultsDataHead = NULL;
  results->resultsDataTail = NULL;

  *ptrResults = results;
}

unsigned long *getPtrSizesToTest(struct testResults *results) {
  return results->sizesToTest;
}

void createResultDataAndAddToTestResults(struct resultsData **ptrData,
                                         struct testResults *results,
                                         const char *resultsName,
                                         bool printOnlyInVerbose,
                                         bool reportAsBandwidth) {
  unsigned int i, j;
  struct resultsData *data;
  data = (struct resultsData *)malloc(sizeof(struct resultsData));
  memset(data, 0, sizeof(struct resultsData));
  strcpy(data->resultsName, resultsName);
  data->results = results;
  for (i = 0; i < MEMALLOC_TYPE_COUNT; i++) {
    data->runTimesInMs[i] =
        (double **)malloc(results->numSizesToTest * sizeof(double *));
    for (j = 0; j < results->numSizesToTest; j++) {
      data->runTimesInMs[i][j] =
          (double *)malloc(results->numMeasurements * sizeof(double));
    }
    data->averageRunTimesInMs[i] =
        (double *)malloc(results->numSizesToTest * sizeof(double));
    data->stdDevRunTimesInMs[i] =
        (double *)malloc(results->numSizesToTest * sizeof(double));
    data->stdDevBandwidthInMBps[i] =
        (double *)malloc(results->numSizesToTest * sizeof(double));
  }
  data->printOnlyInVerbose = printOnlyInVerbose;
  data->reportAsBandwidth = reportAsBandwidth;
  data->next = NULL;
  *ptrData = data;
  if (results->resultsDataHead == NULL) {
    results->resultsDataHead = data;
    results->resultsDataTail = data;
  } else {
    results->resultsDataTail->next = data;
    results->resultsDataTail = data;
  }
}

double *getPtrRunTimesInMs(struct resultsData *data, int allocType,
                           int sizeIndex) {
  return data->runTimesInMs[allocType][sizeIndex];
}

void freeTestResultsAndAllResultsData(struct testResults *results) {
  struct resultsData *data, *dataToFree;
  unsigned int i, j;
  for (data = results->resultsDataHead; data != NULL;) {
    for (i = 0; i < MEMALLOC_TYPE_COUNT; i++) {
      for (j = 0; j < results->numSizesToTest; j++) {
        free(data->runTimesInMs[i][j]);
      }
      free(data->runTimesInMs[i]);
      free(data->averageRunTimesInMs[i]);
      free(data->stdDevRunTimesInMs[i]);
      free(data->stdDevBandwidthInMBps[i]);
    }
    dataToFree = data;
    data = data->next;
    free(dataToFree);
  }
  free(results->sizesToTest);
  free(results);
}

void calculateAverageAndStdDev(double *pAverage, double *pStdDev,
                               double *allResults, unsigned int count) {
  unsigned int i;
  double average = 0.0;
  double stdDev = 0.0;
  for (i = 0; i < count; i++) {
    average += allResults[i];
  }
  average /= count;
  for (i = 0; i < count; i++) {
    stdDev += (allResults[i] - average) * (allResults[i] - average);
  }
  stdDev /= count;
  stdDev = sqrt(stdDev);
  *pAverage = average;
  *pStdDev = (average == 0.0) ? 0.0 : ((100.0 * stdDev) / average);
}

void calculateStdDevBandwidth(double *pStdDev, double *allResults,
                              unsigned int count, unsigned long size) {
  unsigned int i;
  double bandwidth;
  double average = 0.0;
  double stdDev = 0.0;
  for (i = 0; i < count; i++) {
    bandwidth = (1000 * (size / allResults[i])) / ONE_MB;
    average += bandwidth;
  }
  average /= count;
  for (i = 0; i < count; i++) {
    bandwidth = (1000 * (size / allResults[i])) / ONE_MB;
    stdDev += (bandwidth - average) * (bandwidth - average);
  }
  stdDev /= count;
  stdDev = sqrt(stdDev);
  *pStdDev = (average == 0.0) ? 0.0 : ((100.0 * stdDev) / average);
}

void printTimesInTableFormat(struct testResults *results,
                             struct resultsData *data, bool printAverage,
                             bool printStdDev) {
  unsigned int i, j;
  bool printStdDevBandwidth = printStdDev && data->reportAsBandwidth;
  printf("Size_KB");
  for (i = 0; i < MEMALLOC_TYPE_COUNT; i++) {
    printf("\t%7s", memAllocTypeShortStr[i]);
  }
  printf("\n");
  for (j = 0; j < results->numSizesToTest; j++) {
    printf("%lu", results->sizesToTest[j] / ONE_KB);
    for (i = 0; i < MEMALLOC_TYPE_COUNT; i++) {
      printf(data->reportAsBandwidth ? "\t%7.2lf" : "\t%7.3lf",
             printStdDevBandwidth
                 ? data->stdDevBandwidthInMBps[i][j]
                 : getTimeOrBandwidth(
                       printAverage ? data->averageRunTimesInMs[i][j]
                                    : data->stdDevRunTimesInMs[i][j],
                       results->sizesToTest[j], data->reportAsBandwidth));
    }
    printf("\n");
  }
}

void printAllResultsInVerboseMode(struct testResults *results,
                                  struct resultsData *data) {
  unsigned int i, j, k;
  for (i = 0; i < MEMALLOC_TYPE_COUNT; i++) {
    printf("Verbose mode, printing all results for %s\n", memAllocTypeStr[i]);
    printf("Instance");
    for (j = 0; j < results->numSizesToTest; j++) {
      printf("\t%lu", results->sizesToTest[j] / ONE_KB);
    }
    printf("\n");
    for (k = 0; k < results->numMeasurements; k++) {
      printf("%u", k);
      for (j = 0; j < results->numSizesToTest; j++) {
        printf(data->reportAsBandwidth ? "\t%7.2lf" : "\t%7.3lf",
               getTimeOrBandwidth(data->runTimesInMs[i][j][k],
                                  results->sizesToTest[j],
                                  data->reportAsBandwidth));
      }
      printf("\n");
    }
  }
}

void printResults(struct testResults *results,
                  bool print_launch_transfer_results,
                  bool print_std_deviation) {
  char vulcanPrint[256];
  char resultNameNoSpaces[64];
  unsigned int i, j, k;
  struct resultsData *resultsIter;
  bool sizeGreaterThan1MB;
  for (resultsIter = results->resultsDataHead; resultsIter != NULL;
       resultsIter = resultsIter->next) {
    if (!verboseResults && resultsIter->printOnlyInVerbose) {
      continue;
    }
    if (!print_launch_transfer_results) {
      if (!(strcmp(resultsIter->resultsName, "Overall Time") == 0)) {
        continue;
      }
    }
    // regular print
    printf("\n%s For %s ", resultsIter->resultsName, results->testName);
    printf("\n");
    for (j = 0; j < results->numSizesToTest; j++) {
      for (i = 0; i < MEMALLOC_TYPE_COUNT; i++) {
        calculateAverageAndStdDev(&resultsIter->averageRunTimesInMs[i][j],
                                  &resultsIter->stdDevRunTimesInMs[i][j],
                                  resultsIter->runTimesInMs[i][j],
                                  results->numMeasurements);
        if (resultsIter->reportAsBandwidth) {
          calculateStdDevBandwidth(&resultsIter->stdDevBandwidthInMBps[i][j],
                                   resultsIter->runTimesInMs[i][j],
                                   results->numMeasurements,
                                   results->sizesToTest[j]);
        }
      }
    }
    printf("\nPrinting Average of %u measurements in (%s)\n",
           results->numMeasurements,
           resultsIter->reportAsBandwidth ? UNITS_BW : UNITS_Time);
    printTimesInTableFormat(results, resultsIter, true, false);
    if (print_std_deviation) {
      printf(
          "\nPrinting Standard Deviation as %% of average of %u measurements\n",
          results->numMeasurements);
      printTimesInTableFormat(results, resultsIter, false, true);
    }
    if (verboseResults) {
      printAllResultsInVerboseMode(results, resultsIter);
    }
  }
}
