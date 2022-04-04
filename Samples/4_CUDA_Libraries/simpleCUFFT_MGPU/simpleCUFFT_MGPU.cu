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

/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

// System includes
#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>

//CUFFT Header file
#include <cufftXt.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

// Complex data type
typedef float2 Complex;

static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMulAndScale(cufftComplex *,
                                                   cufftComplex *, int, float);

// Kernel for GPU
void multiplyCoefficient(cudaLibXtDesc *, cudaLibXtDesc *, int, float, int);

// Filtering functions
void Convolve(const Complex *, int, const Complex *, int, Complex *);

// Padding functions
int PadData(const Complex *, Complex **, int, const Complex *, Complex **, int);

////////////////////////////////////////////////////////////////////////////////
// Data configuration
// The filter size is assumed to be a number smaller than the signal size
///////////////////////////////////////////////////////////////////////////////
const int SIGNAL_SIZE = 1018;
const int FILTER_KERNEL_SIZE = 11;
const int GPU_COUNT = 2;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("\n[simpleCUFFT_MGPU] is starting...\n\n");

  int GPU_N;
  checkCudaErrors(cudaGetDeviceCount(&GPU_N));

  if (GPU_N < GPU_COUNT) {
    printf("No. of GPU on node %d\n", GPU_N);
    printf("Two GPUs are required to run simpleCUFFT_MGPU sample code\n");
    exit(EXIT_WAIVED);
  }

  int *major_minor = (int *)malloc(sizeof(int) * GPU_N * 2);
  int found2IdenticalGPUs = 0;
  int nGPUs = 2;
  int *whichGPUs;
  whichGPUs = (int *)malloc(sizeof(int) * nGPUs);

  for (int i = 0; i < GPU_N; i++) {
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, i));
    major_minor[i * 2] = deviceProp.major;
    major_minor[i * 2 + 1] = deviceProp.minor;
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", i,
           deviceProp.name, deviceProp.major, deviceProp.minor);
  }

  for (int i = 0; i < GPU_N; i++) {
    for (int j = i + 1; j < GPU_N; j++) {
      if ((major_minor[i * 2] == major_minor[j * 2]) &&
          (major_minor[i * 2 + 1] == major_minor[j * 2 + 1])) {
        whichGPUs[0] = i;
        whichGPUs[1] = j;
        found2IdenticalGPUs = 1;
        break;
      }
    }
    if (found2IdenticalGPUs) {
      break;
    }
  }

  free(major_minor);
  if (!found2IdenticalGPUs) {
    printf(
        "No Two GPUs with same architecture found\nWaiving simpleCUFFT_2d_MGPU "
        "sample\n");
    exit(EXIT_WAIVED);
  }

  // Allocate host memory for the signal
  Complex *h_signal = (Complex *)malloc(sizeof(Complex) * SIGNAL_SIZE);

  // Initialize the memory for the signal
  for (int i = 0; i < SIGNAL_SIZE; ++i) {
    h_signal[i].x = rand() / (float)RAND_MAX;
    h_signal[i].y = 0;
  }

  // Allocate host memory for the filter
  Complex *h_filter_kernel =
      (Complex *)malloc(sizeof(Complex) * FILTER_KERNEL_SIZE);

  // Initialize the memory for the filter
  for (int i = 0; i < FILTER_KERNEL_SIZE; ++i) {
    h_filter_kernel[i].x = rand() / (float)RAND_MAX;
    h_filter_kernel[i].y = 0;
  }

  // Pad signal and filter kernel
  Complex *h_padded_signal;
  Complex *h_padded_filter_kernel;
  int new_size =
      PadData(h_signal, &h_padded_signal, SIGNAL_SIZE, h_filter_kernel,
              &h_padded_filter_kernel, FILTER_KERNEL_SIZE);

  // cufftCreate() - Create an empty plan
  cufftResult result;
  cufftHandle plan_input;
  checkCudaErrors(cufftCreate(&plan_input));

  // cufftXtSetGPUs() - Define which GPUs to use
  result = cufftXtSetGPUs(plan_input, nGPUs, whichGPUs);

  if (result == CUFFT_INVALID_DEVICE) {
    printf("This sample requires two GPUs on the same board.\n");
    printf("No such board was found. Waiving sample.\n");
    exit(EXIT_WAIVED);
  } else if (result != CUFFT_SUCCESS) {
    printf("cufftXtSetGPUs failed\n");
    exit(EXIT_FAILURE);
  }

  // Print the device information to run the code
  printf("\nRunning on GPUs\n");
  for (int i = 0; i < nGPUs; i++) {
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, whichGPUs[i]));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n",
           whichGPUs[i], deviceProp.name, deviceProp.major, deviceProp.minor);
  }

  size_t *worksize;
  worksize = (size_t *)malloc(sizeof(size_t) * nGPUs);

  // cufftMakePlan1d() - Create the plan
  checkCudaErrors(
      cufftMakePlan1d(plan_input, new_size, CUFFT_C2C, 1, worksize));

  // cufftXtMalloc() - Malloc data on multiple GPUs
  cudaLibXtDesc *d_signal;
  checkCudaErrors(cufftXtMalloc(plan_input, (cudaLibXtDesc **)&d_signal,
                                CUFFT_XT_FORMAT_INPLACE));
  cudaLibXtDesc *d_out_signal;
  checkCudaErrors(cufftXtMalloc(plan_input, (cudaLibXtDesc **)&d_out_signal,
                                CUFFT_XT_FORMAT_INPLACE));
  cudaLibXtDesc *d_filter_kernel;
  checkCudaErrors(cufftXtMalloc(plan_input, (cudaLibXtDesc **)&d_filter_kernel,
                                CUFFT_XT_FORMAT_INPLACE));
  cudaLibXtDesc *d_out_filter_kernel;
  checkCudaErrors(cufftXtMalloc(plan_input,
                                (cudaLibXtDesc **)&d_out_filter_kernel,
                                CUFFT_XT_FORMAT_INPLACE));

  // cufftXtMemcpy() - Copy data from host to multiple GPUs
  checkCudaErrors(cufftXtMemcpy(plan_input, d_signal, h_padded_signal,
                                CUFFT_COPY_HOST_TO_DEVICE));
  checkCudaErrors(cufftXtMemcpy(plan_input, d_filter_kernel,
                                h_padded_filter_kernel,
                                CUFFT_COPY_HOST_TO_DEVICE));

  // cufftXtExecDescriptorC2C() - Execute FFT on data on multiple GPUs
  checkCudaErrors(
      cufftXtExecDescriptorC2C(plan_input, d_signal, d_signal, CUFFT_FORWARD));
  checkCudaErrors(cufftXtExecDescriptorC2C(plan_input, d_filter_kernel,
                                           d_filter_kernel, CUFFT_FORWARD));

  // cufftXtMemcpy() - Copy the data to natural order on GPUs
  checkCudaErrors(cufftXtMemcpy(plan_input, d_out_signal, d_signal,
                                CUFFT_COPY_DEVICE_TO_DEVICE));
  checkCudaErrors(cufftXtMemcpy(plan_input, d_out_filter_kernel,
                                d_filter_kernel, CUFFT_COPY_DEVICE_TO_DEVICE));

  printf("\n\nValue of Library Descriptor\n");
  printf("Number of GPUs %d\n", d_out_signal->descriptor->nGPUs);
  printf("Device id  %d %d\n", d_out_signal->descriptor->GPUs[0],
         d_out_signal->descriptor->GPUs[1]);
  printf("Data size on GPU %ld %ld\n",
         (long)(d_out_signal->descriptor->size[0] / sizeof(cufftComplex)),
         (long)(d_out_signal->descriptor->size[1] / sizeof(cufftComplex)));

  // Multiply the coefficients together and normalize the result
  printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
  multiplyCoefficient(d_out_signal, d_out_filter_kernel, new_size,
                      1.0f / new_size, nGPUs);

  // cufftXtExecDescriptorC2C() - Execute inverse  FFT on data on multiple GPUs
  printf("Transforming signal back cufftExecC2C\n");
  checkCudaErrors(cufftXtExecDescriptorC2C(plan_input, d_out_signal,
                                           d_out_signal, CUFFT_INVERSE));

  // Create host pointer pointing to padded signal
  Complex *h_convolved_signal = h_padded_signal;

  // Allocate host memory for the convolution result
  Complex *h_convolved_signal_ref =
      (Complex *)malloc(sizeof(Complex) * SIGNAL_SIZE);

  // cufftXtMemcpy() - Copy data from multiple GPUs to host
  checkCudaErrors(cufftXtMemcpy(plan_input, h_convolved_signal, d_out_signal,
                                CUFFT_COPY_DEVICE_TO_HOST));

  // Convolve on the host
  Convolve(h_signal, SIGNAL_SIZE, h_filter_kernel, FILTER_KERNEL_SIZE,
           h_convolved_signal_ref);

  // Compare CPU and GPU result
  bool bTestResult =
      sdkCompareL2fe((float *)h_convolved_signal_ref,
                     (float *)h_convolved_signal, 2 * SIGNAL_SIZE, 1e-5f);
  printf("\nvalue of TestResult %d\n", bTestResult);

  // Cleanup memory
  free(whichGPUs);
  free(worksize);
  free(h_signal);
  free(h_filter_kernel);
  free(h_padded_signal);
  free(h_padded_filter_kernel);
  free(h_convolved_signal_ref);

  // cudaXtFree() - Free GPU memory
  checkCudaErrors(cufftXtFree(d_signal));
  checkCudaErrors(cufftXtFree(d_filter_kernel));
  checkCudaErrors(cufftXtFree(d_out_signal));
  checkCudaErrors(cufftXtFree(d_out_filter_kernel));

  // cufftDestroy() - Destroy FFT plan
  checkCudaErrors(cufftDestroy(plan_input));

  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

///////////////////////////////////////////////////////////////////////////////////
// Function for padding original data
//////////////////////////////////////////////////////////////////////////////////
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
// Filtering operations - Computing Convolution on the host
////////////////////////////////////////////////////////////////////////////////
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
//  Launch Kernel on multiple GPU
////////////////////////////////////////////////////////////////////////////////
void multiplyCoefficient(cudaLibXtDesc *d_signal,
                         cudaLibXtDesc *d_filter_kernel, int new_size,
                         float val, int nGPUs) {
  int device;
  // Launch the ComplexPointwiseMulAndScale<<< >>> kernel on multiple GPU
  for (int i = 0; i < nGPUs; i++) {
    device = d_signal->descriptor->GPUs[i];

    // Set device
    checkCudaErrors(cudaSetDevice(device));

    // Perform GPU computations
    ComplexPointwiseMulAndScale<<<32, 256>>>(
        (cufftComplex *)d_signal->descriptor->data[i],
        (cufftComplex *)d_filter_kernel->descriptor->data[i],
        int(d_signal->descriptor->size[i] / sizeof(cufftComplex)), val);
  }

  // Wait for device to finish all operation
  for (int i = 0; i < nGPUs; i++) {
    device = d_signal->descriptor->GPUs[i];
    checkCudaErrors(cudaSetDevice(device));
    cudaDeviceSynchronize();
    // Check if kernel execution generated and error
    getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");
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
// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(cufftComplex *a,
                                                   cufftComplex *b, int size,
                                                   float scale) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size; i += numThreads) {
    a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
  }
}
