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

#include "../inc/test.h"

#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <cassert>
#include <typeinfo>
#include <stdio.h>
#include <helper_timer.h>
#include <cuda_runtime.h>
#include <math.h>

#include "../inc/piestimator.h"

template <typename Real>
bool Test<Real>::operator()() {
  using std::endl;
  using std::setw;
  using std::stringstream;

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);

  // Get device properties
  struct cudaDeviceProp deviceProperties;
  cudaError_t cudaResult = cudaGetDeviceProperties(&deviceProperties, device);

  if (cudaResult != cudaSuccess) {
    std::string msg("Could not get device properties: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Evaluate on GPU
  printf("Estimating Pi on GPU (%s)\n\n", deviceProperties.name);
  PiEstimator<Real> estimator(numSims, device, threadBlockSize, seed);
  sdkStartTimer(&timer);
  Real result = estimator();
  sdkStopTimer(&timer);
  elapsedTime = sdkGetAverageTimerValue(&timer) / 1000.0f;

  // Tolerance to compare result with expected
  // This is just to check that nothing has gone very wrong with the
  // test, the actual accuracy of the result depends on the number of
  // Monte Carlo trials
  const Real tolerance = static_cast<Real>(0.01);

  // Display results
  Real abserror = fabs(result - static_cast<float>(PI));
  Real relerror = abserror / static_cast<float>(PI);
  printf("Precision:      %s\n",
         (typeid(Real) == typeid(double)) ? "double" : "single");
  printf("Number of sims: %d\n", numSims);
  printf("Tolerance:      %e\n", tolerance);
  printf("GPU result:     %e\n", result);
  printf("Expected:       %e\n", PI);
  printf("Absolute error: %e\n", abserror);
  printf("Relative error: %e\n\n", relerror);

  // Check result
  if (relerror > tolerance) {
    printf("computed result (%e) does not match expected result (%e).\n",
           result, PI);
    pass = false;
  } else {
    pass = true;
  }

  // Print results
  printf(
      "MonteCarloEstimatePiP, Performance = %.2f sims/s, Time = %.2f(ms), "
      "NumDevsUsed = %u, Blocksize = %u\n",
      numSims / elapsedTime, elapsedTime * 1000.0f, 1, threadBlockSize);

  return pass;
}

// Explicit template instantiation
template struct Test<float>;
template struct Test<double>;
