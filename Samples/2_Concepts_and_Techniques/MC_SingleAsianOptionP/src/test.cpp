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
#include <cuda_runtime.h>
#include <math.h>

#include <helper_timer.h>

#include "../inc/asianoption.h"
#include "../inc/pricingengine.h"

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

  // This test prices a single Asian call option with European
  // exercise, with the priced averaged arithmetically on discrete
  // trading days (weekdays).
  AsianOption<Real> option;
  option.spot = static_cast<Real>(40);
  option.strike = static_cast<Real>(35);
  option.r = static_cast<Real>(0.03);
  option.sigma = static_cast<Real>(0.20);
  option.tenor = static_cast<Real>(1.0 / 3.0);
  option.dt = static_cast<Real>(1.0 / 261);
  option.type = AsianOption<Real>::Call;
  option.value = static_cast<Real>(0.0);
  option.golden = static_cast<Real>(5.162534);

  // Evaluate on GPU
  printf("Pricing option on GPU (%s)\n\n", deviceProperties.name);
  PricingEngine<Real> pricer(numSims, device, threadBlockSize, seed);
  sdkStartTimer(&timer);
  pricer(option);
  sdkStopTimer(&timer);
  elapsedTime = sdkGetAverageTimerValue(&timer) / 1000.0f;

  // Tolerance to compare result with expected
  // This is just to check that nothing has gone very wrong with the
  // test, the actual accuracy of the result depends on the number of
  // Monte Carlo trials
  const Real tolerance = static_cast<Real>(0.1);

  // Display results
  stringstream output;
  output << "Precision:      "
         << ((typeid(Real) == typeid(double)) ? "double" : "single") << endl;
  output << "Number of sims: " << numSims << endl;
  output << endl;
  output << "   Spot    |   Strike   |     r      |   sigma    |   tenor    |  "
            "Call/Put  |   Value    |  Expected  |"
         << endl;
  output << "-----------|------------|------------|------------|------------|--"
            "----------|------------|------------|"
         << endl;
  output << setw(10) << option.spot << " | ";
  output << setw(10) << option.strike << " | ";
  output << setw(10) << option.r << " | ";
  output << setw(10) << option.sigma << " | ";
  output << setw(10) << option.tenor << " | ";
  output << setw(10)
         << (option.type == AsianOption<Real>::Call ? "Call" : "Put") << " | ";
  output << setw(10) << option.value << " | ";
  output << setw(10) << option.golden << " |";

  printf("%s\n\n", output.str().c_str());

  // Check result
  if (fabs(option.value - option.golden) > tolerance) {
    printf("computed result (%e) does not match expected result (%e).\n",
           option.value, option.golden);
    pass = false;
  } else {
    pass = true;
  }

  // Print results
  printf(
      "MonteCarloSingleAsianOptionP, Performance = %.2f sims/s, Time = "
      "%.2f(ms), NumDevsUsed = %u, Blocksize = %u\n",
      numSims / elapsedTime, elapsedTime * 1000.0f, 1, threadBlockSize);

  sdkDeleteTimer(&timer);

  return pass;
}

// Explicit template instantiation
template struct Test<float>;
template struct Test<double>;
