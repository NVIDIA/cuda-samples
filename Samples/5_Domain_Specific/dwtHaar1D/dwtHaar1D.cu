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
* 1D DWT for Haar wavelet and signals with a length which is a power of 2.
* The code reduces bank conflicts and non-coalesced reads / writes as
* appropriate but does not fully remove them because the computational
* overhead to achieve this would outweighs the benefit (see inline comments
* for more details).
* Large signals are subdivided into sub-signals with 512 elements and the
* wavelet transform for these is computed with one block over 10 decomposition
* levels. The resulting signal consisting of the approximation coefficients at
* level X is then processed in a subsequent step on the device. This requires
* interblock synchronization which is only possible on host side.
* Detail coefficients which have been computed are not further referenced
* during the decomposition so that they can be stored directly in their final
* position in global memory. The transform and its storing scheme preserve
* locality in the coefficients so that these writes are coalesced.
* Approximation coefficients are stored in shared memory because they are
* needed to compute the subsequent decomposition step. The top most
* approximation coefficient for a sub-signal processed by one block is stored
* in a special global memory location to simplify the processing after the
* interblock synchronization.
* Most books on wavelets explain the Haar wavelet decomposition. A good freely
* available resource is the Wavelet primer by Stollnitz et al.
* http://grail.cs.washington.edu/projects/wavelets/article/wavelet1.pdf
* http://grail.cs.washington.edu/projects/wavelets/article/wavelet2.pdf
* The basic of all Wavelet transforms is to decompose a signal into
* approximation (a) and detail (d) coefficients where the detail tends to be
* small or zero which allows / simplifies compression. The following "graphs"
* demonstrate the transform for a signal
* of length eight. The index always describes the decomposition level where
* a coefficient arises. The input signal is interpreted as approximation signal
* at level 0. The coefficients computed on the device are stored in the same
* scheme as in the example. This data structure is particularly well suited for
* compression and also preserves the hierarchical structure of the
decomposition.

-------------------------------------------------
| a_0 | a_0 | a_0 | a_0 | a_0 | a_0 | a_0 | a_0 |
-------------------------------------------------

-------------------------------------------------
| a_1 | a_1 | a_1 | a_1 | d_1 | d_1 | d_1 | d_1 |
-------------------------------------------------

-------------------------------------------------
| a_2 | a_2 | d_2 | d_2 | d_1 | d_1 | d_1 | d_1 |
-------------------------------------------------

-------------------------------------------------
| a_3 | d_3 | d_2 | d_2 | d_1 | d_1 | d_1 | d_1 |
-------------------------------------------------

* Host code.
*/

#ifdef _WIN32
#define NOMINMAX
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include <helper_functions.h>
#include <helper_cuda.h>

// constants which are used in host and device code
#define INV_SQRT_2 0.70710678118654752440f;
const unsigned int LOG_NUM_BANKS = 4;
const unsigned int NUM_BANKS = 16;

////////////////////////////////////////////////////////////////////////////////
// includes, kernels
#include "dwtHaar1D_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);
bool getLevels(unsigned int len, unsigned int *levels);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // run test
  runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Perform the wavelet decomposition
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  bool bResult = false;  // flag for final validation of the results

  char *s_fname = NULL, *r_gold_fname = NULL;
  char r_fname[256];
  const char usage[] = {
      "\nUsage:\n"
      "  dwtHaar1D --signal=<signal_file> --result=<result_file> "
      "--gold=<gold_file>\n\n"
      "  <signal_file> Input file containing the signal\n"
      "  <result_file> Output file storing the result of the wavelet "
      "decomposition\n"
      "  <gold_file>   Input file containing the reference result of the "
      "wavelet decomposition\n"
      "\nExample:\n"
      "  ./dwtHaar1D\n"
      "       --signal=signal.dat\n"
      "       --result=result.dat\n"
      "       --gold=regression.gold.dat\n"};

  printf("%s Starting...\n\n", argv[0]);

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  findCudaDevice(argc, (const char **)argv);

  // file names, either specified as cmd line args or use default
  if (argc == 4) {
    char *tmp_sfname, *tmp_rfname, *tmp_goldfname;

    if ((getCmdLineArgumentString(argc, (const char **)argv, "signal",
                                  &tmp_sfname) != true) ||
        (getCmdLineArgumentString(argc, (const char **)argv, "result",
                                  &tmp_rfname) != true) ||
        (getCmdLineArgumentString(argc, (const char **)argv, "gold",
                                  &tmp_goldfname) != true)) {
      fprintf(stderr, "Invalid input syntax.\n%s", usage);
      exit(EXIT_FAILURE);
    }

    s_fname = sdkFindFilePath(tmp_sfname, argv[0]);
    r_gold_fname = sdkFindFilePath(tmp_goldfname, argv[0]);
    strcpy(r_fname, tmp_rfname);
  } else {
    s_fname = sdkFindFilePath("signal.dat", argv[0]);
    r_gold_fname = sdkFindFilePath("regression.gold.dat", argv[0]);
    strcpy(r_fname, "result.dat");
  }

  printf("source file    = \"%s\"\n", s_fname);
  printf("reference file = \"%s\"\n", r_fname);
  printf("gold file      = \"%s\"\n", r_gold_fname);

  // read in signal
  unsigned int slength = 0;
  float *signal = NULL;

  if (s_fname == NULL) {
    fprintf(stderr, "Cannot find the file containing the signal.\n%s", usage);

    exit(EXIT_FAILURE);
  }

  if (sdkReadFile(s_fname, &signal, &slength, false) == true) {
    printf("Reading signal from \"%s\"\n", s_fname);
  } else {
    exit(EXIT_FAILURE);
  }

  // get the number of decompositions necessary to perform a full decomposition
  unsigned int dlevels_complete = 0;

  if (true != getLevels(slength, &dlevels_complete)) {
    // error message
    fprintf(stderr, "Signal length not supported.\n");
    // cleanup and abort
    free(signal);
    exit(EXIT_FAILURE);
  }

  // device in data
  float *d_idata = NULL;
  // device out data
  float *d_odata = NULL;
  // device approx_final data
  float *approx_final = NULL;
  // The very final approximation coefficient has to be written to the output
  // data, all others are reused as input data in the next global step and
  // therefore have to be written to the input data again.
  // The following flag indicates where to copy approx_final data
  //   - 0 is input, 1 is output
  int approx_is_input;

  // allocate device mem
  const unsigned int smem_size = sizeof(float) * slength;
  checkCudaErrors(cudaMalloc((void **)&d_idata, smem_size));
  checkCudaErrors(cudaMalloc((void **)&d_odata, smem_size));
  checkCudaErrors(cudaMalloc((void **)&approx_final, smem_size));
  // copy input data to device
  checkCudaErrors(
      cudaMemcpy(d_idata, signal, smem_size, cudaMemcpyHostToDevice));

  // total number of threads
  // in the first decomposition step always one thread computes the average and
  // detail signal for one pair of adjacent values
  unsigned int num_threads_total_left = slength / 2;
  // decomposition levels performed in the current / next step
  unsigned int dlevels_step = dlevels_complete;

  // 1D signal so the arrangement of elements is also 1D
  dim3 block_size;
  dim3 grid_size;

  // number of decomposition levels left after one iteration on the device
  unsigned int dlevels_left = dlevels_complete;

  // if less or equal 1k elements, then the data can be processed in one block,
  // this avoids the Wait-For-Idle (WFI) on host side which is necessary if the
  // computation is split across multiple SM's if enough input data
  if (dlevels_complete <= 10) {
    // decomposition can be performed at once
    block_size.x = num_threads_total_left;
    approx_is_input = 0;
  } else {
    // 512 threads per block
    grid_size.x = (num_threads_total_left / 512);
    block_size.x = 512;

    // 512 threads corresponds to 10 decomposition steps
    dlevels_step = 10;
    dlevels_left -= 10;

    approx_is_input = 1;
  }

  // Initialize d_odata to 0.0f
  initValue<<<grid_size, block_size>>>(d_odata, 0.0f);

  // do until full decomposition is accomplished
  while (0 != num_threads_total_left) {
    // double the number of threads as bytes
    unsigned int mem_shared = (2 * block_size.x) * sizeof(float);
    // extra memory requirements to avoid bank conflicts
    mem_shared += ((2 * block_size.x) / NUM_BANKS) * sizeof(float);

    // run kernel
    dwtHaar1D<<<grid_size, block_size, mem_shared>>>(
        d_idata, d_odata, approx_final, dlevels_step, num_threads_total_left,
        block_size.x);

    // Copy approx_final to appropriate location
    if (approx_is_input) {
      checkCudaErrors(cudaMemcpy(d_idata, approx_final, grid_size.x * 4,
                                 cudaMemcpyDeviceToDevice));
    } else {
      checkCudaErrors(cudaMemcpy(d_odata, approx_final, grid_size.x * 4,
                                 cudaMemcpyDeviceToDevice));
    }

    // update level variables
    if (dlevels_left < 10) {
      // approx_final = d_odata;
      approx_is_input = 0;
    }

    // more global steps necessary
    dlevels_step = (dlevels_left > 10) ? dlevels_left - 10 : dlevels_left;
    dlevels_left -= 10;

    // after each step only half the threads are used any longer
    // therefore after 10 steps 2^10 less threads
    num_threads_total_left = num_threads_total_left >> 10;

    // update block and grid size
    grid_size.x =
        (num_threads_total_left / 512) + (0 != (num_threads_total_left % 512))
            ? 1
            : 0;

    if (grid_size.x <= 1) {
      block_size.x = num_threads_total_left;
    }
  }

  // get the result back from the server
  // allocate mem for the result
  float *odata = (float *)malloc(smem_size);
  checkCudaErrors(
      cudaMemcpy(odata, d_odata, smem_size, cudaMemcpyDeviceToHost));

  // post processing
  // write file for regression test
  if (r_fname == NULL) {
    fprintf(stderr,
            "Cannot write the output file storing the result of the wavelet "
            "decomposition.\n%s",
            usage);
    exit(EXIT_FAILURE);
  }

  if (sdkWriteFile(r_fname, odata, slength, 0.001f, false) == true) {
    printf("Writing result to \"%s\"\n", r_fname);
  } else {
    exit(EXIT_FAILURE);
  }

  // load the reference solution
  unsigned int len_reference = 0;
  float *reference = NULL;

  if (r_gold_fname == NULL) {
    fprintf(stderr,
            "Cannot read the file containing the reference result of the "
            "wavelet decomposition.\n%s",
            usage);

    exit(EXIT_FAILURE);
  }

  if (sdkReadFile(r_gold_fname, &reference, &len_reference, false) == true) {
    printf("Reading reference result from \"%s\"\n", r_gold_fname);
  } else {
    exit(EXIT_FAILURE);
  }

  assert(slength == len_reference);

  // compare the computed solution and the reference
  bResult = (bool)sdkCompareL2fe(reference, odata, slength, 0.001f);
  free(reference);

  // free allocated host and device memory
  checkCudaErrors(cudaFree(d_odata));
  checkCudaErrors(cudaFree(d_idata));
  checkCudaErrors(cudaFree(approx_final));

  free(signal);
  free(odata);
  free(s_fname);
  free(r_gold_fname);

  printf(bResult ? "Test success!\n" : "Test failure!\n");
}

////////////////////////////////////////////////////////////////////////////////
//! Get number of decomposition levels to perform a full decomposition
//! Also check if the input signal size is suitable
//! @return  true if the number of decomposition levels could be determined
//!          and the signal length is supported by the implementation,
//!          otherwise false
//! @param   len  length of input signal
//! @param   levels  number of decomposition levels necessary to perform a full
//!           decomposition
////////////////////////////////////////////////////////////////////////////////
bool getLevels(unsigned int len, unsigned int *levels) {
  bool retval = false;

  // currently signals up to a length of 2^20 supported
  for (unsigned int i = 0; i < 20; ++i) {
    if (len == (1 << i)) {
      *levels = i;
      retval = true;
      break;
    }
  }

  return retval;
}
