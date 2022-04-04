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

#include "FDTD3d.h"

#include <iostream>
#include <iomanip>

#include "FDTD3dReference.h"
#include "FDTD3dGPU.h"

#include <helper_functions.h>

#include <math.h>
#include <assert.h>

#ifndef CLAMP
#define CLAMP(a, min, max) (MIN(max, MAX(a, min)))
#endif

//// Name of the log file
// const char *printfFile = "FDTD3d.txt";

// Forward declarations
bool runTest(int argc, const char **argv);
void showHelp(const int argc, const char **argv);

int main(int argc, char **argv) {
  bool bTestResult = false;
  // Start the log
  printf("%s Starting...\n\n", argv[0]);

  // Check help flag
  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printf("Displaying help on console\n");
    showHelp(argc, (const char **)argv);
    bTestResult = true;
  } else {
    // Execute
    bTestResult = runTest(argc, (const char **)argv);
  }

  // Finish
  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void showHelp(const int argc, const char **argv) {
  if (argc > 0) std::cout << std::endl << argv[0] << std::endl;

  std::cout << std::endl << "Syntax:" << std::endl;
  std::cout << std::left;
  std::cout << "    " << std::setw(20) << "--device=<device>"
            << "Specify device to use for execution" << std::endl;
  std::cout << "    " << std::setw(20) << "--dimx=<N>"
            << "Specify number of elements in x direction (excluding halo)"
            << std::endl;
  std::cout << "    " << std::setw(20) << "--dimy=<N>"
            << "Specify number of elements in y direction (excluding halo)"
            << std::endl;
  std::cout << "    " << std::setw(20) << "--dimz=<N>"
            << "Specify number of elements in z direction (excluding halo)"
            << std::endl;
  std::cout << "    " << std::setw(20) << "--radius=<N>"
            << "Specify radius of stencil" << std::endl;
  std::cout << "    " << std::setw(20) << "--timesteps=<N>"
            << "Specify number of timesteps" << std::endl;
  std::cout << "    " << std::setw(20) << "--block-size=<N>"
            << "Specify number of threads per block" << std::endl;
  std::cout << std::endl;
  std::cout << "    " << std::setw(20) << "--noprompt"
            << "Skip prompt before exit" << std::endl;
  std::cout << std::endl;
}

bool runTest(int argc, const char **argv) {
  float *host_output;
  float *device_output;
  float *input;
  float *coeff;

  int defaultDim;
  int dimx;
  int dimy;
  int dimz;
  int outerDimx;
  int outerDimy;
  int outerDimz;
  int radius;
  int timesteps;
  size_t volumeSize;
  memsize_t memsize;

  const float lowerBound = 0.0f;
  const float upperBound = 1.0f;

  // Determine default dimensions
  printf("Set-up, based upon target device GMEM size...\n");
  // Get the memory size of the target device
  printf(" getTargetDeviceGlobalMemSize\n");
  getTargetDeviceGlobalMemSize(&memsize, argc, argv);

  // We can never use all the memory so to keep things simple we aim to
  // use around half the total memory
  memsize /= 2;

  // Most of our memory use is taken up by the input and output buffers -
  // two buffers of equal size - and for simplicity the volume is a cube:
  //   dim = floor( (N/2)^(1/3) )
  defaultDim = (int)floor(pow((memsize / (2.0 * sizeof(float))), 1.0 / 3.0));

  // By default, make the volume edge size an integer multiple of 128B to
  // improve performance by coalescing memory accesses, in a real
  // application it would make sense to pad the lines accordingly
  int roundTarget = 128 / sizeof(float);
  defaultDim = defaultDim / roundTarget * roundTarget;
  defaultDim -= k_radius_default * 2;

  // Check dimension is valid
  if (defaultDim < k_dim_min) {
    printf(
        "insufficient device memory (maximum volume on device is %d, must be "
        "between %d and %d).\n",
        defaultDim, k_dim_min, k_dim_max);
    exit(EXIT_FAILURE);
  } else if (defaultDim > k_dim_max) {
    defaultDim = k_dim_max;
  }

  // For QA testing, override default volume size
  if (checkCmdLineFlag(argc, argv, "qatest")) {
    defaultDim = MIN(defaultDim, k_dim_qa);
  }

  // set default dim
  dimx = defaultDim;
  dimy = defaultDim;
  dimz = defaultDim;
  radius = k_radius_default;
  timesteps = k_timesteps_default;

  // Parse command line arguments
  if (checkCmdLineFlag(argc, argv, "dimx")) {
    dimx =
        CLAMP(getCmdLineArgumentInt(argc, argv, "dimx"), k_dim_min, k_dim_max);
  }

  if (checkCmdLineFlag(argc, argv, "dimy")) {
    dimy =
        CLAMP(getCmdLineArgumentInt(argc, argv, "dimy"), k_dim_min, k_dim_max);
  }

  if (checkCmdLineFlag(argc, argv, "dimz")) {
    dimz =
        CLAMP(getCmdLineArgumentInt(argc, argv, "dimz"), k_dim_min, k_dim_max);
  }

  if (checkCmdLineFlag(argc, argv, "radius")) {
    radius = CLAMP(getCmdLineArgumentInt(argc, argv, "radius"), k_radius_min,
                   k_radius_max);
  }

  if (checkCmdLineFlag(argc, argv, "timesteps")) {
    timesteps = CLAMP(getCmdLineArgumentInt(argc, argv, "timesteps"),
                      k_timesteps_min, k_timesteps_max);
  }

  // Determine volume size
  outerDimx = dimx + 2 * radius;
  outerDimy = dimy + 2 * radius;
  outerDimz = dimz + 2 * radius;
  volumeSize = outerDimx * outerDimy * outerDimz;

  // Allocate memory
  host_output = (float *)calloc(volumeSize, sizeof(float));
  input = (float *)malloc(volumeSize * sizeof(float));
  coeff = (float *)malloc((radius + 1) * sizeof(float));

  // Create coefficients
  for (int i = 0; i <= radius; i++) {
    coeff[i] = 0.1f;
  }

  // Generate data
  printf(" generateRandomData\n\n");
  generateRandomData(input, outerDimx, outerDimy, outerDimz, lowerBound,
                     upperBound);
  printf(
      "FDTD on %d x %d x %d volume with symmetric filter radius %d for %d "
      "timesteps...\n\n",
      dimx, dimy, dimz, radius, timesteps);

  // Execute on the host
  printf("fdtdReference...\n");
  fdtdReference(host_output, input, coeff, dimx, dimy, dimz, radius, timesteps);
  printf("fdtdReference complete\n");

  // Allocate memory
  device_output = (float *)calloc(volumeSize, sizeof(float));

  // Execute on the device
  printf("fdtdGPU...\n");
  fdtdGPU(device_output, input, coeff, dimx, dimy, dimz, radius, timesteps,
          argc, argv);
  printf("fdtdGPU complete\n");

  // Compare the results
  float tolerance = 0.0001f;
  printf("\nCompareData (tolerance %f)...\n", tolerance);
  return compareData(device_output, host_output, dimx, dimy, dimz, radius,
                     tolerance);
}
