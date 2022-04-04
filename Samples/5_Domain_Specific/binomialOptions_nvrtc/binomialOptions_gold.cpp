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
#include <math.h>
#include "binomialOptions_common.h"

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////

static double CND(double d) {
  const double A1 = 0.31938153;
  const double A2 = -0.356563782;
  const double A3 = 1.781477937;
  const double A4 = -1.821255978;
  const double A5 = 1.330274429;
  const double RSQRT2PI = 0.39894228040143267793994605993438;

  double K = 1.0 / (1.0 + 0.2316419 * fabs(d));

  double cnd = RSQRT2PI * exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0) cnd = 1.0 - cnd;

  return cnd;
}

extern "C" void BlackScholesCall(float &callResult, TOptionData optionData) {
  double S = optionData.S;
  double X = optionData.X;
  double T = optionData.T;
  double R = optionData.R;
  double V = optionData.V;
  double sqrtT = sqrt(T);

  double d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
  double d2 = d1 - V * sqrtT;

  double CNDD1 = CND(d1);
  double CNDD2 = CND(d2);

  // Calculate Call and Put simultaneously
  double expRT = exp(-R * T);

  callResult = (float)(S * CNDD1 - X * expRT * CNDD2);
}

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on CPU
// Note that CPU code is for correctness testing only and not for benchmarking.
////////////////////////////////////////////////////////////////////////////////

static double expiryCallValue(double S, double X, double vDt, int i) {
  double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
  return (d > 0) ? d : 0;
}

extern "C" void binomialOptionsCPU(float &callResult, TOptionData optionData) {
  static double Call[NUM_STEPS + 1];
  const double S = optionData.S;
  const double X = optionData.X;
  const double T = optionData.T;
  const double R = optionData.R;
  const double V = optionData.V;

  const double dt = T / (double)NUM_STEPS;
  const double vDt = V * sqrt(dt);
  const double rDt = R * dt;

  // Per-step interest and discount factors
  const double If = exp(rDt);
  const double Df = exp(-rDt);

  // Values and pseudoprobabilities of upward and downward moves
  const double u = exp(vDt);
  const double d = exp(-vDt);
  const double pu = (If - d) / (u - d);
  const double pd = 1.0 - pu;
  const double puByDf = pu * Df;
  const double pdByDf = pd * Df;

  ///////////////////////////////////////////////////////////////////////
  // Compute values at expiration date:
  // call option value at period end is V(T) = S(T) - X
  // if S(T) is greater than X, or zero otherwise.
  // The computation is similar for put options.
  ///////////////////////////////////////////////////////////////////////

  for (int i = 0; i <= NUM_STEPS; i++) Call[i] = expiryCallValue(S, X, vDt, i);

  ////////////////////////////////////////////////////////////////////////
  // Walk backwards up binomial tree
  ////////////////////////////////////////////////////////////////////////

  for (int i = NUM_STEPS; i > 0; i--)
    for (int j = 0; j <= i - 1; j++)
      Call[j] = puByDf * Call[j + 1] + pdByDf * Call[j];

  callResult = (float)Call[0];
}
