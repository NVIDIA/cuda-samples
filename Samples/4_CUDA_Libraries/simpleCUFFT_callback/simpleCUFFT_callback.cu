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
 * Example showing the use of CUFFT for fast 1D-convolution using FFT. 
 * This sample is the same as simpleCUFFT, except that it uses a callback
 * function to perform the pointwise multiply and scale, on input to the
 * inverse transform.
 * 
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_functions.h>
#include <helper_cuda.h>

// Complex data type
typedef float2 Complex;
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);

// This is the callback routine prototype
static __device__ cufftComplex ComplexPointwiseMulAndScale(void *a,
                                                           size_t index,
                                                           void *cb_info,
                                                           void *sharedmem);

typedef struct _cb_params {
  Complex *filter;
  float scale;
} cb_params;

// This is the callback routine. It does complex pointwise multiplication with
// scaling.
static __device__ cufftComplex ComplexPointwiseMulAndScale(void *a,
                                                           size_t index,
                                                           void *cb_info,
                                                           void *sharedmem) {
  cb_params *my_params = (cb_params *)cb_info;
  return (cufftComplex)ComplexScale(
      ComplexMul(((Complex *)a)[index], (my_params->filter)[index]),
      my_params->scale);
}

// Define the device pointer to the callback routine. The host code will fetch
// this and pass it to CUFFT
__device__ cufftCallbackLoadC myOwnCallbackPtr = ComplexPointwiseMulAndScale;
// Filtering functions
void Convolve(const Complex *, int, const Complex *, int, Complex *);

// Padding functions
int PadData(const Complex *, Complex **, int, const Complex *, Complex **, int);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest(int argc, char **argv);

// The filter size is assumed to be a number smaller than the signal size
#define SIGNAL_SIZE 50
#define FILTER_KERNEL_SIZE 11

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  struct cudaDeviceProp properties;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&properties, device));
  if (!(properties.major >= 2)) {
    printf("simpleCUFFT_callback requires CUDA architecture SM2.0 or higher\n");
    return EXIT_WAIVED;
  }

  return runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUFFT callbacks
////////////////////////////////////////////////////////////////////////////////
int runTest(int argc, char **argv) {
  printf("[simpleCUFFT_callback] is starting...\n");

  findCudaDevice(argc, (const char **)argv);

  // Allocate host memory for the signal
  Complex *h_signal = (Complex *)malloc(sizeof(Complex) * SIGNAL_SIZE);

  // Initialize the memory for the signal
  for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
    h_signal[i].x = rand() / (float)RAND_MAX;
    h_signal[i].y = 0;
  }

  // Allocate host memory for the filter
  Complex *h_filter_kernel =
      (Complex *)malloc(sizeof(Complex) * FILTER_KERNEL_SIZE);

  // Initialize the memory for the filter
  for (unsigned int i = 0; i < FILTER_KERNEL_SIZE; ++i) {
    h_filter_kernel[i].x = rand() / (float)RAND_MAX;
    h_filter_kernel[i].y = 0;
  }

  // Pad signal and filter kernel
  Complex *h_padded_signal;
  Complex *h_padded_filter_kernel;
  int new_size =
      PadData(h_signal, &h_padded_signal, SIGNAL_SIZE, h_filter_kernel,
              &h_padded_filter_kernel, FILTER_KERNEL_SIZE);
  int mem_size = sizeof(Complex) * new_size;

  // Allocate device memory for signal
  Complex *d_signal;
  checkCudaErrors(cudaMalloc((void **)&d_signal, mem_size));
  // Copy host memory to device
  checkCudaErrors(
      cudaMemcpy(d_signal, h_padded_signal, mem_size, cudaMemcpyHostToDevice));

  // Allocate device memory for filter kernel
  Complex *d_filter_kernel;
  checkCudaErrors(cudaMalloc((void **)&d_filter_kernel, mem_size));

  // Copy host memory to device
  checkCudaErrors(cudaMemcpy(d_filter_kernel, h_padded_filter_kernel, mem_size,
                             cudaMemcpyHostToDevice));

  // Create one CUFFT plan for the forward transforms, and one for the reverse
  // transform with load callback.
  cufftHandle plan, cb_plan;
  size_t work_size;

  checkCudaErrors(cufftCreate(&plan));
  checkCudaErrors(cufftCreate(&cb_plan));

  checkCudaErrors(cufftMakePlan1d(plan, new_size, CUFFT_C2C, 1, &work_size));
  checkCudaErrors(cufftMakePlan1d(cb_plan, new_size, CUFFT_C2C, 1, &work_size));

  // Define a structure used to pass in the device address of the filter kernel,
  // and the scale factor
  cb_params h_params;

  h_params.filter = d_filter_kernel;
  h_params.scale = 1.0f / new_size;

  // Allocate device memory for parameters
  cb_params *d_params;
  checkCudaErrors(cudaMalloc((void **)&d_params, sizeof(cb_params)));

  // Copy host memory to device
  checkCudaErrors(cudaMemcpy(d_params, &h_params, sizeof(cb_params),
                             cudaMemcpyHostToDevice));

  // The host needs to get a copy of the device pointer to the callback
  cufftCallbackLoadC hostCopyOfCallbackPtr;

  checkCudaErrors(cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr, myOwnCallbackPtr,
                                       sizeof(hostCopyOfCallbackPtr)));

  // Now associate the load callback with the plan.
  cufftResult status =
      cufftXtSetCallback(cb_plan, (void **)&hostCopyOfCallbackPtr,
                         CUFFT_CB_LD_COMPLEX, (void **)&d_params);
  if (status == CUFFT_LICENSE_ERROR) {
    printf("This sample requires a valid license file.\n");
    printf(
        "The file was either not found, out of date, or otherwise invalid.\n");
    return EXIT_WAIVED;
  }

  checkCudaErrors(cufftXtSetCallback(cb_plan, (void **)&hostCopyOfCallbackPtr,
                                     CUFFT_CB_LD_COMPLEX, (void **)&d_params));

  // Transform signal and kernel
  printf("Transforming signal cufftExecC2C\n");
  checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal,
                               (cufftComplex *)d_signal, CUFFT_FORWARD));
  checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_filter_kernel,
                               (cufftComplex *)d_filter_kernel, CUFFT_FORWARD));

  // Transform signal back, using the callback to do the pointwise multiply on
  // the way in.
  printf("Transforming signal back cufftExecC2C\n");
  checkCudaErrors(cufftExecC2C(cb_plan, (cufftComplex *)d_signal,
                               (cufftComplex *)d_signal, CUFFT_INVERSE));

  // Copy device memory to host
  Complex *h_convolved_signal = h_padded_signal;
  checkCudaErrors(cudaMemcpy(h_convolved_signal, d_signal, mem_size,
                             cudaMemcpyDeviceToHost));

  // Allocate host memory for the convolution result
  Complex *h_convolved_signal_ref =
      (Complex *)malloc(sizeof(Complex) * SIGNAL_SIZE);

  // Convolve on the host
  Convolve(h_signal, SIGNAL_SIZE, h_filter_kernel, FILTER_KERNEL_SIZE,
           h_convolved_signal_ref);

  // check result
  bool bTestResult =
      sdkCompareL2fe((float *)h_convolved_signal_ref,
                     (float *)h_convolved_signal, 2 * SIGNAL_SIZE, 1e-5f);

  // Destroy CUFFT context
  checkCudaErrors(cufftDestroy(plan));
  checkCudaErrors(cufftDestroy(cb_plan));

  // cleanup memory
  free(h_signal);
  free(h_filter_kernel);
  free(h_padded_signal);
  free(h_padded_filter_kernel);
  free(h_convolved_signal_ref);
  checkCudaErrors(cudaFree(d_signal));
  checkCudaErrors(cudaFree(d_filter_kernel));
  checkCudaErrors(cudaFree(d_params));

  return bTestResult ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Pad data
int PadData(const Complex *signal, Complex **padded_signal, int signal_size,
            const Complex *filter_kernel, Complex **padded_filter_kernel,
            int filter_kernel_size) {
  int minRadius = filter_kernel_size / 2;
  int maxRadius = filter_kernel_size - minRadius;
  int new_size = signal_size + maxRadius;

  // Pad signal
  Complex *new_data = (Complex *)malloc(sizeof(Complex) * new_size);
  memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
  memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
  *padded_signal = new_data;

  // Pad filter
  new_data = (Complex *)malloc(sizeof(Complex) * new_size);
  memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
  memset(new_data + maxRadius, 0,
         (new_size - filter_kernel_size) * sizeof(Complex));
  memcpy(new_data + new_size - minRadius, filter_kernel,
         minRadius * sizeof(Complex));
  *padded_filter_kernel = new_data;

  return new_size;
}

////////////////////////////////////////////////////////////////////////////////
// Filtering operations
////////////////////////////////////////////////////////////////////////////////

// Computes convolution on the host
void Convolve(const Complex *signal, int signal_size,
              const Complex *filter_kernel, int filter_kernel_size,
              Complex *filtered_signal) {
  int minRadius = filter_kernel_size / 2;
  int maxRadius = filter_kernel_size - minRadius;

  // Loop over output element indices
  for (int i = 0; i < signal_size; ++i) {
    filtered_signal[i].x = filtered_signal[i].y = 0;

    // Loop over convolution indices
    for (int j = -maxRadius + 1; j <= minRadius; ++j) {
      int k = i + j;

      if (k >= 0 && k < signal_size) {
        filtered_signal[i] =
            ComplexAdd(filtered_signal[i],
                       ComplexMul(signal[k], filter_kernel[minRadius - j]));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
  Complex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}
