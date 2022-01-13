/**
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Helper functions for CUDA Driver API error handling (make sure that CUDA_H is
// included in your projects)
#ifndef HELPER_CUDA_DRVAPI_H
#define HELPER_CUDA_DRVAPI_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <helper_string.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#ifndef HELPER_CUDA_DRVAPI_H
inline int ftoi(float value) {
  return (value >= 0 ? static_cast<int>(value + 0.5)
                     : static_cast<int>(value - 0.5));
}
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// add a level of protection to the CUDA SDK samples, let's force samples to
// explicitly include CUDA.H
#ifdef __cuda_cuda_h__
// This will output the proper CUDA error strings in the event that a CUDA host
// call returns an error
#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    const char *errorStr = NULL;
    cuGetErrorString(err, &errorStr);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}
#endif

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
                             int device) {
  checkCudaErrors(cuDeviceGetAttribute(attribute, device_attribute, device));
}
#endif

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2CoresDRV(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine the #
  // of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m = SM
             // minor version
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
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one to run
  // properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}
  // end of GPU Architecture definitions

#ifdef __cuda_cuda_h__
// General GPU Device CUDA Initialization
inline int gpuDeviceInitDRV(int ARGC, const char **ARGV) {
  int cuDevice = 0;
  int deviceCount = 0;
  checkCudaErrors(cuInit(0, __CUDA_API_VERSION));

  checkCudaErrors(cuDeviceGetCount(&deviceCount));

  if (deviceCount == 0) {
    fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
    exit(EXIT_FAILURE);
  }

  int dev = 0;
  dev = getCmdLineArgumentInt(ARGC, (const char **)ARGV, "device=");

  if (dev < 0) {
    dev = 0;
  }

  if (dev > deviceCount - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
            deviceCount);
    fprintf(stderr,
            ">> cudaDeviceInit (-device=%d) is not a valid GPU device. <<\n",
            dev);
    fprintf(stderr, "\n");
    return -dev;
  }

  checkCudaErrors(cuDeviceGet(&cuDevice, dev));
  char name[100];
  checkCudaErrors(cuDeviceGetName(name, 100, cuDevice));

  int computeMode;
  getCudaAttribute<int>(&computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev);

  if (computeMode == CU_COMPUTEMODE_PROHIBITED) {
    fprintf(stderr,
            "Error: device is running in <CU_COMPUTEMODE_PROHIBITED>, no "
            "threads can use this CUDA Device.\n");
    return -1;
  }

  if (checkCmdLineFlag(ARGC, (const char **)ARGV, "quiet") == false) {
    printf("gpuDeviceInitDRV() Using CUDA Device [%d]: %s\n", dev, name);
  }

  return dev;
}

// This function returns the best GPU based on performance
inline int gpuGetMaxGflopsDeviceIdDRV() {
  CUdevice current_device = 0;
  CUdevice max_perf_device = 0;
  int device_count = 0;
  int sm_per_multiproc = 0;
  unsigned long long max_compute_perf = 0;
  int major = 0;
  int minor = 0;
  int multiProcessorCount;
  int clockRate;
  int devices_prohibited = 0;

  cuInit(0, __CUDA_API_VERSION);
  checkCudaErrors(cuDeviceGetCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceIdDRV error: no devices supporting CUDA\n");
    exit(EXIT_FAILURE);
  }

  // Find the best CUDA capable GPU device
  current_device = 0;

  while (current_device < device_count) {
    checkCudaErrors(cuDeviceGetAttribute(
        &multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        current_device));
    checkCudaErrors(cuDeviceGetAttribute(
        &clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, current_device));
    checkCudaErrors(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, current_device));
    checkCudaErrors(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, current_device));

    int computeMode;
    getCudaAttribute<int>(&computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
                          current_device);

    if (computeMode != CU_COMPUTEMODE_PROHIBITED) {
      if (major == 9999 && minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc = _ConvertSMVer2CoresDRV(major, minor);
      }

      unsigned long long compute_perf =
          (unsigned long long)(multiProcessorCount * sm_per_multiproc *
                               clockRate);

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
            "gpuGetMaxGflopsDeviceIdDRV error: all devices have compute mode "
            "prohibited.\n");
    exit(EXIT_FAILURE);
  }

  return max_perf_device;
}

// General initialization call to pick the best CUDA Device
inline CUdevice findCudaDeviceDRV(int argc, const char **argv) {
  CUdevice cuDevice;
  int devID = 0;

  // If the command-line has a device number specified, use it
  if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
    devID = gpuDeviceInitDRV(argc, argv);

    if (devID < 0) {
      printf("exiting...\n");
      exit(EXIT_SUCCESS);
    }
  } else {
    // Otherwise pick the device with highest Gflops/s
    char name[100];
    devID = gpuGetMaxGflopsDeviceIdDRV();
    checkCudaErrors(cuDeviceGet(&cuDevice, devID));
    cuDeviceGetName(name, 100, cuDevice);
    printf("> Using CUDA Device [%d]: %s\n", devID, name);
  }

  cuDeviceGet(&cuDevice, devID);

  return cuDevice;
}

inline CUdevice findIntegratedGPUDrv() {
  CUdevice current_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;
  int isIntegrated;

  cuInit(0, __CUDA_API_VERSION);
  checkCudaErrors(cuDeviceGetCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the integrated GPU which is compute capable
  while (current_device < device_count) {
    int computeMode = -1;
    checkCudaErrors(cuDeviceGetAttribute(
        &isIntegrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, current_device));
    checkCudaErrors(cuDeviceGetAttribute(
        &computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, current_device));

    // If GPU is integrated and is not running on Compute Mode prohibited use
    // that
    if (isIntegrated && (computeMode != CU_COMPUTEMODE_PROHIBITED)) {
      int major = 0, minor = 0;
      char deviceName[256];
      checkCudaErrors(cuDeviceGetAttribute(
          &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
          current_device));
      checkCudaErrors(cuDeviceGetAttribute(
          &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
          current_device));
      checkCudaErrors(cuDeviceGetName(deviceName, 256, current_device));
      printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
             current_device, deviceName, major, minor);

      return current_device;
    } else {
      devices_prohibited++;
    }

    current_device++;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr, "CUDA error: No Integrated CUDA capable GPU found.\n");
    exit(EXIT_FAILURE);
  }

  return -1;
}

// General check for CUDA GPU SM Capabilities
inline bool checkCudaCapabilitiesDRV(int major_version, int minor_version,
                                     int devID) {
  CUdevice cuDevice;
  char name[256];
  int major = 0, minor = 0;

  checkCudaErrors(cuDeviceGet(&cuDevice, devID));
  checkCudaErrors(cuDeviceGetName(name, 100, cuDevice));
  checkCudaErrors(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  checkCudaErrors(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

  if ((major > major_version) ||
      (major == major_version && minor >= minor_version)) {
    printf("> Device %d: <%16s >, Compute SM %d.%d detected\n", devID, name,
           major, minor);
    return true;
  } else {
    printf(
        "No GPU device was found that can support CUDA compute capability "
        "%d.%d.\n",
        major_version, minor_version);
    return false;
  }
}
#endif

  // end of CUDA Helper Functions

#endif  // HELPER_CUDA_DRVAPI_H

