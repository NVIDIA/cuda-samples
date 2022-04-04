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

///////////////////////////////////////////////////////////////////////////////
// Monte Carlo: Estimate Pi
// ========================
//
// This sample demonstrates a very simple Monte Carlo estimation for Pi.
//
// This file, main.cpp, contains the setup information to run the test, for
// example parsing the command line and integrating this sample with the
// samples framework. As such it is perhaps less interesting than the guts of
// the sample. Readers wishing to skip the clutter are advised to skip straight
// to Test.operator() in test.cpp.
///////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cuda_runtime.h>
#include <helper_timer.h>
#include <helper_cuda.h>

#include <math.h>
#include "../inc/test.h"

// Forward declarations
void showHelp(const int argc, const char **argv);
template <typename Real>
void runTest(int argc, const char **argv);

int main(int argc, char **argv) {
  using std::invalid_argument;
  using std::string;

  // Open the log file
  printf("Monte Carlo Estimate Pi (with inline QRNG)\n");
  printf("==========================================\n\n");

  // If help flag is set, display help and exit immediately
  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printf("Displaying help on console\n");
    showHelp(argc, (const char **)argv);
    exit(EXIT_SUCCESS);
  }

  // Check the precision (checked against the device capability later)
  try {
    char *value;

    if (getCmdLineArgumentString(argc, (const char **)argv, "precision",
                                 &value)) {
      // Check requested precision is valid
      string prec(value);

      if (prec.compare("single") == 0 || prec.compare("\"single\"") == 0) {
        runTest<float>(argc, (const char **)argv);
      } else if (prec.compare("double") == 0 ||
                 prec.compare("\"double\"") == 0) {
        runTest<double>(argc, (const char **)argv);
      } else {
        printf(
            "specified precision (%s) is invalid, must be \"single\" or "
            "\"double\".\n",
            value);
        throw invalid_argument("precision");
      }
    } else {
      runTest<float>(argc, (const char **)argv);
    }
  } catch (invalid_argument &e) {
    printf("invalid command line argument (%s)\n", e.what());
    exit(EXIT_FAILURE);
  }

  // Finish
  exit(EXIT_SUCCESS);
}

template <typename Real>
void runTest(int argc, const char **argv) {
  using std::invalid_argument;
  using std::runtime_error;

  StopWatchInterface *timer = NULL;

  try {
    Test<Real> test;
    int deviceCount = 0;
    cudaError_t cudaResult = cudaSuccess;

    // by default specify GPU Device == 0
    test.device = 0;

    // Get number of available devices
    cudaResult = cudaGetDeviceCount(&deviceCount);

    if (cudaResult != cudaSuccess) {
      printf("could not get device count.\n");
      throw runtime_error("cudaGetDeviceCount");
    }

    // (default parameters)
    test.numSims = k_sims_qa;
    test.threadBlockSize = k_bsize_qa;

    {
      char *value = 0;

      if (getCmdLineArgumentString(argc, argv, "device", &value)) {
        test.device = (int)atoi(value);

        if (test.device >= deviceCount) {
          printf(
              "invalid target device specified on command line (device %d does "
              "not exist).\n",
              test.device);
          throw invalid_argument("device");
        }
      } else {
        test.device = gpuGetMaxGflopsDeviceId();
      }

      if (getCmdLineArgumentString(argc, argv, "sims", &value)) {
        test.numSims = (unsigned int)atoi(value);

        if (test.numSims < k_sims_min || test.numSims > k_sims_max) {
          printf(
              "specified number of simulations (%d) is invalid, must be "
              "between %d and %d.\n",
              test.numSims, k_sims_min, k_sims_max);
          throw invalid_argument("sims");
        }
      } else {
        test.numSims = k_sims_def;
      }

      if (getCmdLineArgumentString(argc, argv, "block-size", &value)) {
        // Determine max threads per block
        cudaDeviceProp deviceProperties;
        cudaResult = cudaGetDeviceProperties(&deviceProperties, test.device);

        if (cudaResult != cudaSuccess) {
          printf("cound not get device properties for device %d.\n",
                 test.device);
          throw runtime_error("cudaGetDeviceProperties");
        }

        // Check requested size is valid
        test.threadBlockSize = (unsigned int)atoi(value);

        if (test.threadBlockSize < k_bsize_min ||
            test.threadBlockSize > static_cast<unsigned int>(
                                       deviceProperties.maxThreadsPerBlock)) {
          printf(
              "specified block size (%d) is invalid, must be between %d and %d "
              "for device %d.\n",
              test.threadBlockSize, k_bsize_min,
              deviceProperties.maxThreadsPerBlock, test.device);
          throw invalid_argument("block-size");
        }

        if (test.threadBlockSize & test.threadBlockSize - 1) {
          printf(
              "specified block size (%d) is invalid, must be a power of two "
              "(see reduction function).\n",
              test.threadBlockSize);
          throw invalid_argument("block-size");
        }
      } else {
        test.threadBlockSize = k_bsize_def;
      }
    }
    // Execute
    test();
  } catch (invalid_argument &e) {
    printf("invalid command line argument (%s)\n", e.what());
    exit(EXIT_FAILURE);
  } catch (runtime_error &e) {
    printf("runtime error (%s)\n", e.what());
    exit(EXIT_FAILURE);
  }
}

void showHelp(int argc, const char **argv) {
  using std::cout;
  using std::endl;
  using std::left;
  using std::setw;

  if (argc > 0) {
    cout << endl << argv[0] << endl;
  }

  cout << endl << "Syntax:" << endl;
  cout << left;
  cout << "    " << setw(20) << "--device=<device>"
       << "Specify device to use for execution" << endl;
  cout << "    " << setw(20) << "--sims=<N>"
       << "Specify number of Monte Carlo simulations" << endl;
  cout << "    " << setw(20) << "--block-size=<N>"
       << "Specify number of threads per block" << endl;
  cout << "    " << setw(20) << "--precision=<P>"
       << "Specify the precision (\"single\" or \"double\")" << endl;
  cout << endl;
  cout << "    " << setw(20) << "--noprompt"
       << "Skip prompt before exit" << endl;
  cout << endl;
}
