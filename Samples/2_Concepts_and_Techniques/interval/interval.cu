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

/* Example of program using the interval_gpu<T> template class and operators:
 * Search for roots of a function using an interval Newton method.
  *
 * Use the command-line argument "--n=<N>" to select which GPU implementation to
 * use,
 * otherwise the naive implementation will be used by default.
 * 0: the naive implementation
 * 1: the optimized implementation
 * 2: the recursive implementation
 *
 */

const static char *sSDKsample = "Interval Computing";

#include <iostream>
#include <stdio.h>
#include "helper_cuda.h"
#include "interval.h"
#include "cuda_interval.h"
#include "cpu_interval.h"

int main(int argc, char *argv[]) {
  int implementation_choice = 0;

  printf("[%s]  starting ...\n\n", sSDKsample);

  if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
    implementation_choice =
        getCmdLineArgumentInt(argc, (const char **)argv, "n");
  }

  // Pick the best GPU available, or if the developer selects one at the command
  // line
  int devID = findCudaDevice(argc, (const char **)argv);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, devID);
  printf("> GPU Device has Compute Capabilities SM %d.%d\n\n", deviceProp.major,
         deviceProp.minor);

  switch (implementation_choice) {
    case 0:
      printf("GPU naive implementation\n");
      break;

    case 1:
      printf("GPU optimized implementation\n");
      break;

    case 2:
      printf("GPU recursive implementation (requires Compute SM 2.0+)\n");
      break;

    default:
      printf("GPU naive implementation\n");
  }

  interval_gpu<T> *d_result;
  int *d_nresults;
  int *h_nresults = new int[THREADS];
  cudaEvent_t start, stop;

  CHECKED_CALL(cudaSetDevice(devID));
  CHECKED_CALL(cudaMalloc((void **)&d_result,
                          THREADS * DEPTH_RESULT * sizeof(*d_result)));
  CHECKED_CALL(cudaMalloc((void **)&d_nresults, THREADS * sizeof(*d_nresults)));
  CHECKED_CALL(cudaEventCreate(&start));
  CHECKED_CALL(cudaEventCreate(&stop));

  // We need L1 cache to store the stack (only applicable to sm_20 and higher)
  CHECKED_CALL(
      cudaFuncSetCacheConfig(test_interval_newton<T>, cudaFuncCachePreferL1));

  // Increase the stack size large enough for the non-inlined and recursive
  // function calls (only applicable to sm_20 and higher)
  CHECKED_CALL(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

  interval_gpu<T> i(0.01f, 4.0f);
  std::cout << "Searching for roots in [" << i.lower() << ", " << i.upper()
            << "]...\n";

  CHECKED_CALL(cudaEventRecord(start, 0));

  for (int it = 0; it < NUM_RUNS; ++it) {
    test_interval_newton<T><<<GRID_SIZE, BLOCK_SIZE>>>(d_result, d_nresults, i,
                                                       implementation_choice);
    CHECKED_CALL(cudaGetLastError());
  }

  CHECKED_CALL(cudaEventRecord(stop, 0));
  CHECKED_CALL(cudaDeviceSynchronize());

  I_CPU *h_result = new I_CPU[THREADS * DEPTH_RESULT];
  CHECKED_CALL(cudaMemcpy(h_result, d_result,
                          THREADS * DEPTH_RESULT * sizeof(*d_result),
                          cudaMemcpyDeviceToHost));
  CHECKED_CALL(cudaMemcpy(h_nresults, d_nresults, THREADS * sizeof(*d_nresults),
                          cudaMemcpyDeviceToHost));

  std::cout << "Found " << h_nresults[0]
            << " intervals that may contain the root(s)\n";
  std::cout.precision(15);

  for (int i = 0; i != h_nresults[0]; ++i) {
    std::cout << " i[" << i << "] ="
              << " [" << h_result[THREADS * i + 0].lower() << ", "
              << h_result[THREADS * i + 0].upper() << "]\n";
  }

  float time;
  CHECKED_CALL(cudaEventElapsedTime(&time, start, stop));
  std::cout << "Number of equations solved: " << THREADS << "\n";
  std::cout << "Time per equation: "
            << 1000000.0f * (time / (float)(THREADS)) / NUM_RUNS << " us\n";

  CHECKED_CALL(cudaEventDestroy(start));
  CHECKED_CALL(cudaEventDestroy(stop));
  CHECKED_CALL(cudaFree(d_result));
  CHECKED_CALL(cudaFree(d_nresults));

  // Compute the results using a CPU implementation based on the Boost library
  I_CPU i_cpu(0.01f, 4.0f);
  I_CPU *h_result_cpu = new I_CPU[THREADS * DEPTH_RESULT];
  int *h_nresults_cpu = new int[THREADS];
  test_interval_newton_cpu<I_CPU>(h_result_cpu, h_nresults_cpu, i_cpu);

  // Compare the CPU and GPU results
  bool bTestResult =
      checkAgainstHost(h_nresults, h_nresults_cpu, h_result, h_result_cpu);

  delete[] h_result_cpu;
  delete[] h_nresults_cpu;
  delete[] h_result;
  delete[] h_nresults;

  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
