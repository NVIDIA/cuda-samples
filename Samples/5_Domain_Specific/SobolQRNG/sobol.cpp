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

/*
 * Portions Copyright (c) 2009 Mike Giles, Oxford University.  All rights
 * reserved.
 * Portions Copyright (c) 2008 Frances Y. Kuo and Stephen Joe.  All rights
 * reserved.
 *
 * Sobol Quasi-random Number Generator example
 *
 * Based on CUDA code submitted by Mike Giles, Oxford University, United Kingdom
 * http://people.maths.ox.ac.uk/~gilesm/
 *
 * and C code developed by Stephen Joe, University of Waikato, New Zealand
 * and Frances Kuo, University of New South Wales, Australia
 * http://web.maths.unsw.edu.au/~fkuo/sobol/
 *
 * For theoretical background see:
 *
 * P. Bratley and B.L. Fox.
 * Implementing Sobol's quasirandom sequence generator
 * http://portal.acm.org/citation.cfm?id=42288
 * ACM Trans. on Math. Software, 14(1):88-100, 1988
 *
 * S. Joe and F. Kuo.
 * Remark on algorithm 659: implementing Sobol's quasirandom sequence generator.
 * http://portal.acm.org/citation.cfm?id=641879
 * ACM Trans. on Math. Software, 29(1):49-57, 2003
 */

#include <iostream>

#include <cuda_runtime.h>  // CUDA Runtime Functions
#include <helper_cuda.h>  // helper functions for CUDA error checking and initialization
#include <helper_functions.h>  // helper functions

#include <stdexcept>
#include <math.h>

#include "sobol.h"
#include "sobol_gold.h"
#include "sobol_gpu.h"

#define L1ERROR_TOLERANCE (1e-6)

const char *sSDKsample = "Sobol Quasi-Random Number Generator";

void printHelp(int argc, char *argv[]) {
  if (argc > 0) {
    std::cout << "\nUsage: " << argv[0] << " <options>\n\n";
  } else {
    std::cout << "\nUsage: <program name> <options>\n\n";
  }

  std::cout << "\t--vectors=M     specify number of vectors    (required)\n";
  std::cout << "\t                The generator will output M vectors\n\n";
  std::cout << "\t--dimensions=N  specify number of dimensions (required)\n";
  std::cout << "\t                Each vector will consist of N components\n\n";
  std::cout << std::endl;
}

int main(int argc, char *argv[]) {
  bool ok = true;

  // We will generate n_vectors vectors of n_dimensions numbers
  int n_vectors = 100000;
  int n_dimensions = 100;

  printf("%s Starting...\n\n", sSDKsample);

  // Print help if requested
  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printHelp(argc, argv);
    return 0;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "qatest")) {
    // For QA testing set a default number of vectors and dimensions
    n_vectors = 100000;
    n_dimensions = 100;
  } else {
    // Parse the command line to determine the required number of vectors
    if (checkCmdLineFlag(argc, (const char **)argv, "vectors")) {
      n_vectors = getCmdLineArgumentInt(argc, (const char **)argv, "vectors");

      if (n_vectors < 1) {
        std::cerr << "Illegal argument: number of vectors must be positive "
                     "(--vectors=N)"
                  << std::endl;
        ok = false;
      }
    }

    std::cout << "> number of vectors = " << n_vectors << std::endl;

    // Parse the command line to determine the number of dimensions in each
    // vector
    if (checkCmdLineFlag(argc, (const char **)argv, "dimensions")) {
      n_dimensions =
          getCmdLineArgumentInt(argc, (const char **)argv, "dimensions");

      if (n_dimensions < 1) {
        std::cerr << "Illegal argument: number of dimensions must be positive "
                     "(--dimensions=N)"
                  << std::endl;
        ok = false;
      }
    }

    std::cout << "> number of dimensions = " << n_dimensions << std::endl;
  }

  // If any of the command line checks failed, exit
  if (!ok) {
    return -1;
  }

  // Use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  findCudaDevice(argc, (const char **)argv);

  // Create a timer to measure performance
  StopWatchInterface *hTimer = NULL;
  double time;
  sdkCreateTimer(&hTimer);

  // Allocate memory for the arrays
  std::cout << "Allocating CPU memory..." << std::endl;
  unsigned int *h_directions = 0;
  float *h_outputCPU = 0;
  float *h_outputGPU = 0;

  try {
    h_directions = new unsigned int[n_dimensions * n_directions];
    h_outputCPU = new float[n_vectors * n_dimensions];
    h_outputGPU = new float[n_vectors * n_dimensions];
  } catch (std::exception e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::cerr << "Unable to allocate CPU memory (try running with fewer "
                 "vectors/dimensions)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Allocating GPU memory..." << std::endl;
  unsigned int *d_directions;
  float *d_output;

  try {
    cudaError_t cudaResult;
    cudaResult = cudaMalloc((void **)&d_directions,
                            n_dimensions * n_directions * sizeof(unsigned int));

    if (cudaResult != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(cudaResult));
    }

    cudaResult = cudaMalloc((void **)&d_output,
                            n_vectors * n_dimensions * sizeof(float));

    if (cudaResult != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(cudaResult));
    }
  } catch (std::runtime_error e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::cerr << "Unable to allocate GPU memory (try running with fewer "
                 "vectors/dimensions)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // Initialize the direction numbers (done on the host)
  std::cout << "Initializing direction numbers..." << std::endl;
  initSobolDirectionVectors(n_dimensions, h_directions);

  // Copy the direction numbers to the device
  std::cout << "Copying direction numbers to device..." << std::endl;
  checkCudaErrors(cudaMemcpy(d_directions, h_directions,
                             n_dimensions * n_directions * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaDeviceSynchronize());

  // Execute the QRNG on the device
  std::cout << "Executing QRNG on GPU..." << std::endl;
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  sobolGPU(n_vectors, n_dimensions, d_directions, d_output);
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  time = sdkGetTimerValue(&hTimer);

  if (time < 1e-6) {
    std::cout << "Gsamples/s: problem size too small to measure, try "
                 "increasing number of vectors or dimensions"
              << std::endl;
  } else {
    std::cout << "Gsamples/s: "
              << (double)n_vectors * (double)n_dimensions * 1E-9 / (time * 1E-3)
              << std::endl;
  }

  std::cout << "Reading results from GPU..." << std::endl;
  checkCudaErrors(cudaMemcpy(h_outputGPU, d_output,
                             n_vectors * n_dimensions * sizeof(float),
                             cudaMemcpyDeviceToHost));

  std::cout << std::endl;
  // Execute the QRNG on the host
  std::cout << "Executing QRNG on CPU..." << std::endl;
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  sobolCPU(n_vectors, n_dimensions, h_directions, h_outputCPU);
  sdkStopTimer(&hTimer);
  time = sdkGetTimerValue(&hTimer);

  if (time < 1e-6) {
    std::cout << "Gsamples/s: problem size too small to measure, try "
                 "increasing number of vectors or dimensions"
              << std::endl;
  } else {
    std::cout << "Gsamples/s: "
              << (double)n_vectors * (double)n_dimensions * 1E-9 / (time * 1E-3)
              << std::endl;
  }

  // Check the results
  std::cout << "Checking results..." << std::endl;
  float l1norm_diff = 0.0F;
  float l1norm_ref = 0.0F;
  float l1error;

  // Special case if n_vectors is 1, when the vector should be exactly 0
  if (n_vectors == 1) {
    for (int d = 0, v = 0; d < n_dimensions; d++) {
      float ref = h_outputCPU[d * n_vectors + v];
      l1norm_diff += fabs(h_outputGPU[d * n_vectors + v] - ref);
      l1norm_ref += fabs(ref);
    }

    // Output the L1-Error
    l1error = l1norm_diff;

    if (l1norm_ref != 0) {
      std::cerr << "Error: L1-Norm of the reference is not zero (for single "
                   "vector), golden generator appears broken\n";
    } else {
      std::cout << "L1-Error: " << l1error << std::endl;
    }
  } else {
    for (int d = 0; d < n_dimensions; d++) {
      for (int v = 0; v < n_vectors; v++) {
        float ref = h_outputCPU[d * n_vectors + v];
        l1norm_diff += fabs(h_outputGPU[d * n_vectors + v] - ref);
        l1norm_ref += fabs(ref);
      }
    }

    // Output the L1-Error
    l1error = l1norm_diff / l1norm_ref;

    if (l1norm_ref == 0) {
      std::cerr << "Error: L1-Norm of the reference is zero, golden generator "
                   "appears broken\n";
    } else {
      std::cout << "L1-Error: " << l1error << std::endl;
    }
  }

  // Cleanup and terminate
  std::cout << "Shutting down..." << std::endl;
  sdkDeleteTimer(&hTimer);
  delete h_directions;
  delete h_outputCPU;
  delete h_outputGPU;
  checkCudaErrors(cudaFree(d_directions));
  checkCudaErrors(cudaFree(d_output));

  // Check pass/fail using L1 error
  exit(l1error < L1ERROR_TOLERANCE ? EXIT_SUCCESS : EXIT_FAILURE);
}
