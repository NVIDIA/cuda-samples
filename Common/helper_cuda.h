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

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking

#ifndef COMMON_HELPER_CUDA_H_
#define COMMON_HELPER_CUDA_H_

#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <helper_string.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// Note, it is required that your SDK sample to include the proper header
// files, please refer the CUDA examples for examples of the needed CUDA
// headers, which may change depending on which CUDA functions are used.

// CUDA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}
#endif

#ifdef CUDA_DRIVER_API
// CUDA Driver API errors
static const char *_cudaGetErrorEnum(CUresult error) {
  static char unknown[] = "<unknown>";
  const char *ret = NULL;
  cuGetErrorName(error, &ret);
  return ret ? ret : unknown;
}
#endif

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef _CUFFT_H_
// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error) {
  switch (error) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";

    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST";

    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";

    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR";

    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";

    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";

    case CUFFT_LICENSE_ERROR:
      return "CUFFT_LICENSE_ERROR";

    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";
  }

  return "<unknown>";
}
#endif

#ifdef CUSPARSEAPI
// cuSPARSE API errors
static const char *_cudaGetErrorEnum(cusparseStatus_t error) {
  switch (error) {
    case CUSPARSE_STATUS_SUCCESS:
      return "CUSPARSE_STATUS_SUCCESS";

    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "CUSPARSE_STATUS_NOT_INITIALIZED";

    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "CUSPARSE_STATUS_ALLOC_FAILED";

    case CUSPARSE_STATUS_INVALID_VALUE:
      return "CUSPARSE_STATUS_INVALID_VALUE";

    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "CUSPARSE_STATUS_ARCH_MISMATCH";

    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "CUSPARSE_STATUS_MAPPING_ERROR";

    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "CUSPARSE_STATUS_EXECUTION_FAILED";

    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "CUSPARSE_STATUS_INTERNAL_ERROR";

    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  }

  return "<unknown>";
}
#endif

#ifdef CUSOLVER_COMMON_H_
// cuSOLVER API errors
static const char *_cudaGetErrorEnum(cusolverStatus_t error) {
  switch (error) {
    case CUSOLVER_STATUS_SUCCESS:
      return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_NOT_SUPPORTED ";
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return "CUSOLVER_STATUS_INVALID_LICENSE";
  }

  return "<unknown>";
}
#endif

#ifdef CURAND_H_
// cuRAND API errors
static const char *_cudaGetErrorEnum(curandStatus_t error) {
  switch (error) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";

    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";

    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";

    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";

    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";

    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";

    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";

    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";

    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";

    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";

    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef NVJPEGAPI
// nvJPEG API errors
static const char *_cudaGetErrorEnum(nvjpegStatus_t error) {
  switch (error) {
    case NVJPEG_STATUS_SUCCESS:
      return "NVJPEG_STATUS_SUCCESS";

    case NVJPEG_STATUS_NOT_INITIALIZED:
      return "NVJPEG_STATUS_NOT_INITIALIZED";

    case NVJPEG_STATUS_INVALID_PARAMETER:
      return "NVJPEG_STATUS_INVALID_PARAMETER";

    case NVJPEG_STATUS_BAD_JPEG:
      return "NVJPEG_STATUS_BAD_JPEG";

    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
      return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";

    case NVJPEG_STATUS_ALLOCATOR_FAILURE:
      return "NVJPEG_STATUS_ALLOCATOR_FAILURE";

    case NVJPEG_STATUS_EXECUTION_FAILED:
      return "NVJPEG_STATUS_EXECUTION_FAILED";

    case NVJPEG_STATUS_ARCH_MISMATCH:
      return "NVJPEG_STATUS_ARCH_MISMATCH";

    case NVJPEG_STATUS_INTERNAL_ERROR:
      return "NVJPEG_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef NV_NPPIDEFS_H
// NPP API errors
static const char *_cudaGetErrorEnum(NppStatus error) {
  switch (error) {
    case NPP_NOT_SUPPORTED_MODE_ERROR:
      return "NPP_NOT_SUPPORTED_MODE_ERROR";

    case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
      return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";

    case NPP_RESIZE_NO_OPERATION_ERROR:
      return "NPP_RESIZE_NO_OPERATION_ERROR";

    case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
      return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

    case NPP_BAD_ARG_ERROR:
      return "NPP_BAD_ARGUMENT_ERROR";

    case NPP_COEFF_ERROR:
      return "NPP_COEFFICIENT_ERROR";

    case NPP_RECT_ERROR:
      return "NPP_RECTANGLE_ERROR";

    case NPP_QUAD_ERROR:
      return "NPP_QUADRANGLE_ERROR";

    case NPP_MEM_ALLOC_ERR:
      return "NPP_MEMORY_ALLOCATION_ERROR";

    case NPP_HISTO_NUMBER_OF_LEVELS_ERROR:
      return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

    case NPP_INVALID_INPUT:
      return "NPP_INVALID_INPUT";

    case NPP_POINTER_ERROR:
      return "NPP_POINTER_ERROR";

    case NPP_WARNING:
      return "NPP_WARNING";

    case NPP_ODD_ROI_WARNING:
      return "NPP_ODD_ROI_WARNING";
#else

    // These are for CUDA 5.5 or higher
    case NPP_BAD_ARGUMENT_ERROR:
      return "NPP_BAD_ARGUMENT_ERROR";

    case NPP_COEFFICIENT_ERROR:
      return "NPP_COEFFICIENT_ERROR";

    case NPP_RECTANGLE_ERROR:
      return "NPP_RECTANGLE_ERROR";

    case NPP_QUADRANGLE_ERROR:
      return "NPP_QUADRANGLE_ERROR";

    case NPP_MEMORY_ALLOCATION_ERR:
      return "NPP_MEMORY_ALLOCATION_ERROR";

    case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
      return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

    case NPP_INVALID_HOST_POINTER_ERROR:
      return "NPP_INVALID_HOST_POINTER_ERROR";

    case NPP_INVALID_DEVICE_POINTER_ERROR:
      return "NPP_INVALID_DEVICE_POINTER_ERROR";
#endif

    case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
      return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";

    case NPP_TEXTURE_BIND_ERROR:
      return "NPP_TEXTURE_BIND_ERROR";

    case NPP_WRONG_INTERSECTION_ROI_ERROR:
      return "NPP_WRONG_INTERSECTION_ROI_ERROR";

    case NPP_NOT_EVEN_STEP_ERROR:
      return "NPP_NOT_EVEN_STEP_ERROR";

    case NPP_INTERPOLATION_ERROR:
      return "NPP_INTERPOLATION_ERROR";

    case NPP_RESIZE_FACTOR_ERROR:
      return "NPP_RESIZE_FACTOR_ERROR";

    case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
      return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

    case NPP_MEMFREE_ERR:
      return "NPP_MEMFREE_ERR";

    case NPP_MEMSET_ERR:
      return "NPP_MEMSET_ERR";

    case NPP_MEMCPY_ERR:
      return "NPP_MEMCPY_ERROR";

    case NPP_MIRROR_FLIP_ERR:
      return "NPP_MIRROR_FLIP_ERR";
#else

    case NPP_MEMFREE_ERROR:
      return "NPP_MEMFREE_ERROR";

    case NPP_MEMSET_ERROR:
      return "NPP_MEMSET_ERROR";

    case NPP_MEMCPY_ERROR:
      return "NPP_MEMCPY_ERROR";

    case NPP_MIRROR_FLIP_ERROR:
      return "NPP_MIRROR_FLIP_ERROR";
#endif

    case NPP_ALIGNMENT_ERROR:
      return "NPP_ALIGNMENT_ERROR";

    case NPP_STEP_ERROR:
      return "NPP_STEP_ERROR";

    case NPP_SIZE_ERROR:
      return "NPP_SIZE_ERROR";

    case NPP_NULL_POINTER_ERROR:
      return "NPP_NULL_POINTER_ERROR";

    case NPP_CUDA_KERNEL_EXECUTION_ERROR:
      return "NPP_CUDA_KERNEL_EXECUTION_ERROR";

    case NPP_NOT_IMPLEMENTED_ERROR:
      return "NPP_NOT_IMPLEMENTED_ERROR";

    case NPP_ERROR:
      return "NPP_ERROR";

    case NPP_SUCCESS:
      return "NPP_SUCCESS";

    case NPP_WRONG_INTERSECTION_QUAD_WARNING:
      return "NPP_WRONG_INTERSECTION_QUAD_WARNING";

    case NPP_MISALIGNED_DST_ROI_WARNING:
      return "NPP_MISALIGNED_DST_ROI_WARNING";

    case NPP_AFFINE_QUAD_INCORRECT_WARNING:
      return "NPP_AFFINE_QUAD_INCORRECT_WARNING";

    case NPP_DOUBLE_SIZE_WARNING:
      return "NPP_DOUBLE_SIZE_WARNING";

    case NPP_WRONG_INTERSECTION_ROI_WARNING:
      return "NPP_WRONG_INTERSECTION_ROI_WARNING";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x6000
    /* These are 6.0 or higher */
    case NPP_LUT_PALETTE_BITSIZE_ERROR:
      return "NPP_LUT_PALETTE_BITSIZE_ERROR";

    case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
      return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";

    case NPP_QUALITY_INDEX_ERROR:
      return "NPP_QUALITY_INDEX_ERROR";

    case NPP_CHANNEL_ORDER_ERROR:
      return "NPP_CHANNEL_ORDER_ERROR";

    case NPP_ZERO_MASK_VALUE_ERROR:
      return "NPP_ZERO_MASK_VALUE_ERROR";

    case NPP_NUMBER_OF_CHANNELS_ERROR:
      return "NPP_NUMBER_OF_CHANNELS_ERROR";

    case NPP_COI_ERROR:
      return "NPP_COI_ERROR";

    case NPP_DIVISOR_ERROR:
      return "NPP_DIVISOR_ERROR";

    case NPP_CHANNEL_ERROR:
      return "NPP_CHANNEL_ERROR";

    case NPP_STRIDE_ERROR:
      return "NPP_STRIDE_ERROR";

    case NPP_ANCHOR_ERROR:
      return "NPP_ANCHOR_ERROR";

    case NPP_MASK_SIZE_ERROR:
      return "NPP_MASK_SIZE_ERROR";

    case NPP_MOMENT_00_ZERO_ERROR:
      return "NPP_MOMENT_00_ZERO_ERROR";

    case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
      return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";

    case NPP_THRESHOLD_ERROR:
      return "NPP_THRESHOLD_ERROR";

    case NPP_CONTEXT_MATCH_ERROR:
      return "NPP_CONTEXT_MATCH_ERROR";

    case NPP_FFT_FLAG_ERROR:
      return "NPP_FFT_FLAG_ERROR";

    case NPP_FFT_ORDER_ERROR:
      return "NPP_FFT_ORDER_ERROR";

    case NPP_SCALE_RANGE_ERROR:
      return "NPP_SCALE_RANGE_ERROR";

    case NPP_DATA_TYPE_ERROR:
      return "NPP_DATA_TYPE_ERROR";

    case NPP_OUT_OFF_RANGE_ERROR:
      return "NPP_OUT_OFF_RANGE_ERROR";

    case NPP_DIVIDE_BY_ZERO_ERROR:
      return "NPP_DIVIDE_BY_ZERO_ERROR";

    case NPP_RANGE_ERROR:
      return "NPP_RANGE_ERROR";

    case NPP_NO_MEMORY_ERROR:
      return "NPP_NO_MEMORY_ERROR";

    case NPP_ERROR_RESERVED:
      return "NPP_ERROR_RESERVED";

    case NPP_NO_OPERATION_WARNING:
      return "NPP_NO_OPERATION_WARNING";

    case NPP_DIVIDE_BY_ZERO_WARNING:
      return "NPP_DIVIDE_BY_ZERO_WARNING";
#endif

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x7000
    /* These are 7.0 or higher */
    case NPP_OVERFLOW_ERROR:
      return "NPP_OVERFLOW_ERROR";

    case NPP_CORRUPTED_DATA_ERROR:
      return "NPP_CORRUPTED_DATA_ERROR";
#endif
  }

  return "<unknown>";
}
#endif

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#ifdef __DRIVER_TYPES_H__
// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// This will only print the proper error string when calling cudaGetLastError
// but not exit program incase error detected.
#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)

inline void __printLastCudaError(const char *errorMessage, const char *file,
                                 const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
  }
}
#endif

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

// Float To Int conversion
inline int ftoi(float value) {
  return (value >= 0 ? static_cast<int>(value + 0.5)
                     : static_cast<int>(value - 0.5));
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x90, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

inline const char* _ConvertSMVer2ArchName(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the GPU Arch name)
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    const char* name;
  } sSMtoArchName;

  sSMtoArchName nGpuArchNameSM[] = {
      {0x30, "Kepler"},
      {0x32, "Kepler"},
      {0x35, "Kepler"},
      {0x37, "Kepler"},
      {0x50, "Maxwell"},
      {0x52, "Maxwell"},
      {0x53, "Maxwell"},
      {0x60, "Pascal"},
      {0x61, "Pascal"},
      {0x62, "Pascal"},
      {0x70, "Volta"},
      {0x72, "Xavier"},
      {0x75, "Turing"},
      {0x80, "Ampere"},
      {0x86, "Ampere"},
      {0x87, "Ampere"},
      {0x90, "Hopper"},
      {-1, "Graphics Device"}};

  int index = 0;

  while (nGpuArchNameSM[index].SM != -1) {
    if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchNameSM[index].name;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoArchName for SM %d.%d is undefined."
      "  Default to use %s\n",
      major, minor, nGpuArchNameSM[index - 1].name);
  return nGpuArchNameSM[index - 1].name;
}
  // end of GPU Architecture definitions

#ifdef __CUDA_RUNTIME_H__
// General GPU Device CUDA Initialization
inline int gpuDeviceInit(int devID) {
  int device_count;
  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuDeviceInit() CUDA error: "
            "no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  if (devID < 0) {
    devID = 0;
  }

  if (devID > device_count - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
            device_count);
    fprintf(stderr,
            ">> gpuDeviceInit (-device=%d) is not a valid"
            " GPU device. <<\n",
            devID);
    fprintf(stderr, "\n");
    return -devID;
  }

  int computeMode = -1, major = 0, minor = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, devID));
  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
  checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
  if (computeMode == cudaComputeModeProhibited) {
    fprintf(stderr,
            "Error: device is running in <Compute Mode "
            "Prohibited>, no threads can use cudaSetDevice().\n");
    return -1;
  }

  if (major < 1) {
    fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaSetDevice(devID));
  printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, _ConvertSMVer2ArchName(major, minor));

  return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_perf_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  uint64_t max_compute_perf = 0;
  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the best CUDA capable GPU device
  current_device = 0;

  while (current_device < device_count) {
    int computeMode = -1, major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));

    // If this GPU is not running on Compute Mode prohibited,
    // then we can add it to the list
    if (computeMode != cudaComputeModeProhibited) {
      if (major == 9999 && minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc =
            _ConvertSMVer2Cores(major,  minor);
      }
      int multiProcessorCount = 0, clockRate = 0;
      checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
      cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
      if (result != cudaSuccess) {
        // If cudaDevAttrClockRate attribute is not supported we
        // set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
        if(result == cudaErrorInvalidValue) {
          clockRate = 1;
        }
        else {
          fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__, __LINE__,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result));
          exit(EXIT_FAILURE);
        }
      }
      uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

      if (compute_perf > max_compute_perf) {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    } else {
      devices_prohibited++;
    }

    ++current_device;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " all devices have compute mode prohibited.\n");
    exit(EXIT_FAILURE);
  }

  return max_perf_device;
}

// Initialization code to find the best CUDA Device
inline int findCudaDevice(int argc, const char **argv) {
  int devID = 0;

  // If the command-line has a device number specified, use it
  if (checkCmdLineFlag(argc, argv, "device")) {
    devID = getCmdLineArgumentInt(argc, argv, "device=");

    if (devID < 0) {
      printf("Invalid command line parameter\n ");
      exit(EXIT_FAILURE);
    } else {
      devID = gpuDeviceInit(devID);

      if (devID < 0) {
        printf("exiting...\n");
        exit(EXIT_FAILURE);
      }
    }
  } else {
    // Otherwise pick the device with highest Gflops/s
    devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(devID));
    int major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
           devID, _ConvertSMVer2ArchName(major, minor), major, minor);

  }

  return devID;
}

inline int findIntegratedGPU() {
  int current_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the integrated GPU which is compute capable
  while (current_device < device_count) {
    int computeMode = -1, integrated = -1;
    checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
    checkCudaErrors(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, current_device));
    // If GPU is integrated and is not running on Compute Mode prohibited,
    // then cuda can map to GLES resource
    if (integrated && (computeMode != cudaComputeModeProhibited)) {
      checkCudaErrors(cudaSetDevice(current_device));

      int major = 0, minor = 0;
      checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
      checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));
      printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
             current_device, _ConvertSMVer2ArchName(major, minor), major, minor);

      return current_device;
    } else {
      devices_prohibited++;
    }

    current_device++;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr,
            "CUDA error:"
            " No GLES-CUDA Interop capable GPU found.\n");
    exit(EXIT_FAILURE);
  }

  return -1;
}

// General check for CUDA GPU SM Capabilities
inline bool checkCudaCapabilities(int major_version, int minor_version) {
  int dev;
  int major = 0, minor = 0;

  checkCudaErrors(cudaGetDevice(&dev));
  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));

  if ((major > major_version) ||
      (major == major_version &&
       minor >= minor_version)) {
    printf("  Device %d: <%16s >, Compute SM %d.%d detected\n", dev,
           _ConvertSMVer2ArchName(major, minor), major, minor);
    return true;
  } else {
    printf(
        "  No GPU device was found that can support "
        "CUDA compute capability %d.%d.\n",
        major_version, minor_version);
    return false;
  }
}
#endif

  // end of CUDA Helper Functions

#endif  // COMMON_HELPER_CUDA_H_
