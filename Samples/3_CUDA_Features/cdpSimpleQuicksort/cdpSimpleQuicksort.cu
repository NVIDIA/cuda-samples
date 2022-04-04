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

#include <iostream>
#include <cstdio>
#include <helper_cuda.h>
#include <helper_string.h>

#define MAX_DEPTH 16
#define INSERTION_SORT 32

////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
__device__ void selection_sort(unsigned int *data, int left, int right) {
  for (int i = left; i <= right; ++i) {
    unsigned min_val = data[i];
    int min_idx = i;

    // Find the smallest value in the range [left, right].
    for (int j = i + 1; j <= right; ++j) {
      unsigned val_j = data[j];

      if (val_j < min_val) {
        min_idx = j;
        min_val = val_j;
      }
    }

    // Swap the values.
    if (i != min_idx) {
      data[min_idx] = data[i];
      data[i] = min_val;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort(unsigned int *data, int left, int right,
                                     int depth) {
  // If we're too deep or there are few elements left, we use an insertion
  // sort...
  if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
    selection_sort(data, left, right);
    return;
  }

  unsigned int *lptr = data + left;
  unsigned int *rptr = data + right;
  unsigned int pivot = data[(left + right) / 2];

  // Do the partitioning.
  while (lptr <= rptr) {
    // Find the next left- and right-hand values to swap
    unsigned int lval = *lptr;
    unsigned int rval = *rptr;

    // Move the left pointer as long as the pointed element is smaller than the
    // pivot.
    while (lval < pivot) {
      lptr++;
      lval = *lptr;
    }

    // Move the right pointer as long as the pointed element is larger than the
    // pivot.
    while (rval > pivot) {
      rptr--;
      rval = *rptr;
    }

    // If the swap points are valid, do the swap!
    if (lptr <= rptr) {
      *lptr++ = rval;
      *rptr-- = lval;
    }
  }

  // Now the recursive part
  int nright = rptr - data;
  int nleft = lptr - data;

  // Launch a new block to sort the left part.
  if (left < (rptr - data)) {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cdp_simple_quicksort<<<1, 1, 0, s>>>(data, left, nright, depth + 1);
    cudaStreamDestroy(s);
  }

  // Launch a new block to sort the right part.
  if ((lptr - data) < right) {
    cudaStream_t s1;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cdp_simple_quicksort<<<1, 1, 0, s1>>>(data, nleft, right, depth + 1);
    cudaStreamDestroy(s1);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void run_qsort(unsigned int *data, unsigned int nitems) {
  // Prepare CDP for the max depth 'MAX_DEPTH'.
  checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

  // Launch on device
  int left = 0;
  int right = nitems - 1;
  std::cout << "Launching kernel on the GPU" << std::endl;
  cdp_simple_quicksort<<<1, 1>>>(data, left, right, 0);
  checkCudaErrors(cudaDeviceSynchronize());
}

////////////////////////////////////////////////////////////////////////////////
// Initialize data on the host.
////////////////////////////////////////////////////////////////////////////////
void initialize_data(unsigned int *dst, unsigned int nitems) {
  // Fixed seed for illustration
  srand(2047);

  // Fill dst with random values
  for (unsigned i = 0; i < nitems; i++) dst[i] = rand() % nitems;
}

////////////////////////////////////////////////////////////////////////////////
// Verify the results.
////////////////////////////////////////////////////////////////////////////////
void check_results(int n, unsigned int *results_d) {
  unsigned int *results_h = new unsigned[n];
  checkCudaErrors(cudaMemcpy(results_h, results_d, n * sizeof(unsigned),
                             cudaMemcpyDeviceToHost));

  for (int i = 1; i < n; ++i)
    if (results_h[i - 1] > results_h[i]) {
      std::cout << "Invalid item[" << i - 1 << "]: " << results_h[i - 1]
                << " greater than " << results_h[i] << std::endl;
      exit(EXIT_FAILURE);
    }

  std::cout << "OK" << std::endl;
  delete[] results_h;
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  int num_items = 128;
  bool verbose = false;

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "h")) {
    std::cerr << "Usage: " << argv[0]
              << " num_items=<num_items>\twhere num_items is the number of "
                 "items to sort"
              << std::endl;
    exit(EXIT_SUCCESS);
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "v")) {
    verbose = true;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "num_items")) {
    num_items = getCmdLineArgumentInt(argc, (const char **)argv, "num_items");

    if (num_items < 1) {
      std::cerr << "ERROR: num_items has to be greater than 1" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // Find/set device and get device properties
  int device = -1;
  cudaDeviceProp deviceProp;
  device = findCudaDevice(argc, (const char **)argv);
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));

  if (!(deviceProp.major > 3 ||
        (deviceProp.major == 3 && deviceProp.minor >= 5))) {
    printf("GPU %d - %s  does not support CUDA Dynamic Parallelism\n Exiting.",
           device, deviceProp.name);
    exit(EXIT_WAIVED);
  }

  // Create input data
  unsigned int *h_data = 0;
  unsigned int *d_data = 0;

  // Allocate CPU memory and initialize data.
  std::cout << "Initializing data:" << std::endl;
  h_data = (unsigned int *)malloc(num_items * sizeof(unsigned int));
  initialize_data(h_data, num_items);

  if (verbose) {
    for (int i = 0; i < num_items; i++)
      std::cout << "Data [" << i << "]: " << h_data[i] << std::endl;
  }

  // Allocate GPU memory.
  checkCudaErrors(
      cudaMalloc((void **)&d_data, num_items * sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(d_data, h_data, num_items * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));

  // Execute
  std::cout << "Running quicksort on " << num_items << " elements" << std::endl;
  run_qsort(d_data, num_items);

  // Check result
  std::cout << "Validating results: ";
  check_results(num_items, d_data);

  free(h_data);
  checkCudaErrors(cudaFree(d_data));

  exit(EXIT_SUCCESS);
}
