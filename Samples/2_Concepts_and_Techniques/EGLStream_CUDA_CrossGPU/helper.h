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

#include "eglstrm_common.h"
#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif
#include <cuda.h>

int parseCmdLine(int argc, char *argv[], TestArgs *args);
void printUsage(void);
int NUMTRIALS = 10;
int profileAPIs = 0;

bool verbose = 0;
bool isCrossDevice = 0;

// Parse the command line options. Returns FAILURE on a parse error, SUCCESS
// otherwise.
int parseCmdLine(int argc, char *argv[], TestArgs *args) {
  int i;

  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-h") == 0) {
      printUsage();
      exit(0);
    } else if (strcmp(argv[i], "-n") == 0) {
      ++i;
      if (sscanf(argv[i], "%d", &NUMTRIALS) != 1 || NUMTRIALS <= 0) {
        printf("Invalid trial count: %s should be > 0\n", argv[i]);
        return -1;
      }
    } else if (strcmp(argv[i], "-profile") == 0) {
      profileAPIs = 1;
    } else if (strcmp(argv[i], "-crossdev") == 0) {
      isCrossDevice = 1;
    } else if (strcmp(argv[i], "-width") == 0) {
      ++i;
      if (sscanf(argv[i], "%d", &WIDTH) != 1 || (WIDTH <= 0)) {
        printf("Width should be greater than 0\n");
        return -1;
      }
    } else if (strcmp(argv[i], "-height") == 0) {
      ++i;
      if (sscanf(argv[i], "%d", &HEIGHT) != 1 || (HEIGHT <= 0)) {
        printf("Width should be greater than 0\n");
        return -1;
      }
    } else if (0 == strcmp(&argv[i][1], "proctype")) {
      ++i;
      if (!strcasecmp(argv[i], "prod")) {
        args->isProducer = 1;
      } else if (!strcasecmp(argv[i], "cons")) {
        args->isProducer = 0;
      } else {
        printf("%s: Bad Process Type: %s\n", __func__, argv[i]);
        return 1;
      }
    } else if (strcmp(argv[i], "-v") == 0) {
      verbose = 1;
    } else {
      printf("Unknown option: %s\n", argv[i]);
      return -1;
    }
  }

  if (isCrossDevice) {
    int deviceCount = 0;

    CUresult error_id = cuInit(0);
    if (error_id != CUDA_SUCCESS) {
      printf("cuInit(0) returned %d\n", error_id);
      printf("Result = FAIL\n");
      exit(EXIT_FAILURE);
    }

    error_id = cuDeviceGetCount(&deviceCount);
    if (error_id != CUDA_SUCCESS) {
      printf("cuDeviceGetCount returned %d\n", (int)error_id);
      printf("Result = FAIL\n");
      exit(EXIT_FAILURE);
    }

    int iGPUexists = 0;
    CUdevice dev;
    for (dev = 0; dev < deviceCount; ++dev) {
      int integrated = 0;
      CUresult error_result = cuDeviceGetAttribute(
          &integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev);

      if (error_result != CUDA_SUCCESS) {
        printf("cuDeviceGetAttribute returned error : %d\n", (int)error_result);
        exit(EXIT_FAILURE);
      }

      if (integrated) {
        iGPUexists = 1;
      }
    }

    if (!iGPUexists) {
      printf("No Integrated GPU found in the system.\n");
      printf(
          "-crossdev option is only supported on systems with an Integrated "
          "GPU and a Discrete GPU\n");
      printf("Waiving the execution\n");
      exit(EXIT_SUCCESS);
    }
  }

  if (!eglSetupExtensions(isCrossDevice)) {
    printf("SetupExtentions failed \n");
    exit(EXIT_FAILURE);
  }
#define MAX_EGL_DEVICES 4
  EGLDeviceEXT devices[MAX_EGL_DEVICES];
  EGLint numDevices = 0;
  EGLBoolean eglStatus =
      eglQueryDevicesEXT(MAX_EGL_DEVICES, devices, &numDevices);
  if (eglStatus != EGL_TRUE) {
    printf("Error querying EGL devices\n");
    exit(EXIT_FAILURE);
  }

  if (numDevices == 0) {
    printf("No EGL devices found\n");
    eglStatus = EGL_FALSE;
    exit(2);  // EXIT_WAIVED
  }

  int egl_device_id = 0;
  for (egl_device_id = 0; egl_device_id < numDevices; egl_device_id++) {
    EGLAttrib cuda_device;
    eglStatus = eglQueryDeviceAttribEXT(devices[egl_device_id],
                                        EGL_CUDA_DEVICE_NV, &cuda_device);
    if (eglStatus == EGL_TRUE) {
      break;
    }
  }

  if (egl_device_id >= numDevices) {
    printf("No CUDA Capable EGL Device found.. Waiving execution\n");
    exit(2);  // EXIT_WAIVED
  }

  if (isCrossDevice) {
    if (numDevices == 1) {
      printf(
          "Found only one EGL device, cannot setup cross GPU streams. "
          "Waiving\n");
      eglStatus = EGL_FALSE;
      exit(2);  // EXIT_WAIVED
    }
  }

  return 0;
}

void launchProducer(TestArgs *args) {
  /* Cross-process creation of producer */
  char argsProducer[1024];
  char str[256];

  strcpy(argsProducer, "./EGLStream_CUDA_CrossGPU -proctype prod ");

  if (isCrossDevice) {
    sprintf(str, "-crossdev ");
    strcat(argsProducer, str);
  }

  if (verbose) {
    sprintf(str, "-v ");
    strcat(argsProducer, str);
  }

  /*Make the process run in bg*/
  strcat(argsProducer, "& ");

  printf("\n%s: Crossproc Producer command: %s \n", __func__, argsProducer);

  /*Create crossproc Producer*/
  system(argsProducer);

  /*Enable crossproc Consumer in the same process */
  args->isProducer = 0;
}

void printUsage(void) {
  printf("Usage:\n");
  printf("  -h           Print this help message\n");
  printf("  -n n         Exit after running n trials. Set to 10 by default\n");
  printf(
      "  -profile     Profile time taken by ReleaseAPI. Not set by default\n");
  printf("  -crossdev    Run with producer on idgpu and consumer on dgpu\n");
  printf("  -dgpu        (same as -crossdev, deprecated)\n");
  printf("  -v           verbose output\n");
}
