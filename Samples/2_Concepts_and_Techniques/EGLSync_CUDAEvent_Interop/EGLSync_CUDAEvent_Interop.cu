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

// Simple interop app demonstrating EGLImage + EGLSync interop with CUDA.
// Using EGLSync - CUDA Event interop one can achieve synchronization on GPU
// itself for GL-EGL-CUDA operations instead of blocking CPU for
// synchronization. This app requires GLES 3.2 or higher

//---------------------------INCLUDES---------------------------------//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "graphics_interface.h"
#include <cuda.h>
#include <helper_cuda_drvapi.h>
#include <cudaEGL.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include "egl_common.h"

//---------------------------DEFINES---------------------------------//
#define MAX_ITR 100

#define FAILURE 0
#define SUCCESS 1
#define WAIVED 2

#define BLOCK_SIZE 16

#define GL_READ 0
#define GL_WRITE 1
//---------------------------MACROS---------------------------------//

// Error-checking wrapper around GL calls
#define GL_SAFE_CALL(call)                                              \
  {                                                                     \
    GLenum err;                                                         \
    call;                                                               \
    err = glGetError();                                                 \
    if (err != GL_NO_ERROR) {                                           \
      fprintf(stderr, "%s:%d GL error: %d\n", __FILE__, __LINE__, err); \
      cleanup(FAILURE);                                                 \
    }                                                                   \
  }

#define GL_SAFE_CALL_NO_CLEANUP(call, err)                                 \
  {                                                                        \
    GLenum status;                                                         \
    call;                                                                  \
    status = glGetError();                                                 \
    if (status != GL_NO_ERROR) {                                           \
      fprintf(stderr, "%s:%d GL error: %d\n", __FILE__, __LINE__, status); \
      err = status;                                                        \
    }                                                                      \
  }

// Error-checking wrapper around CUDA calls (taken from cutil.h)
#define CUDA_SAFE_CALL(call)                                                  \
  do {                                                                        \
    cudaError err = call;                                                     \
    if (cudaSuccess != err) {                                                 \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, \
              __LINE__, cudaGetErrorString(err));                             \
      cleanup(FAILURE);                                                       \
    }                                                                         \
  } while (0)

#define CUDA_SAFE_CALL_NO_CLEANUP(call, err)                                  \
  do {                                                                        \
    cudaError status = call;                                                  \
    if (cudaSuccess != status) {                                              \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, \
              __LINE__, cudaGetErrorString(status));                          \
      err = status;                                                           \
    }                                                                         \
  } while (0)

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_DECL)
typedef void (*extlst_fnptr_t)(void);
static struct {
  extlst_fnptr_t *fnptr;
  char const *name;
} extensionList[] = {EXTENSION_LIST(EXTLST_ENTRY)};

int eglSetupExtensions(void) {
  unsigned int i;

  for (i = 0; i < (sizeof(extensionList) / sizeof(*extensionList)); i++) {
    *extensionList[i].fnptr = eglGetProcAddress(extensionList[i].name);
    if (*extensionList[i].fnptr == NULL) {
      printf("Couldn't get address of %s()\n", extensionList[i].name);
      return 0;
    }
  }

  return 1;
}
#endif

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif

//------------------------GLOBAL VARIABLES--------------------------//

// GL texture
GLuint tex[2] = {0};

// Used to catch unexpected termination from GLUT
int cleanExit = 0;

// Use CPU Sync or GPU sync; Default GPU
int useGpu = 1;

// CUDA Resource
CUgraphicsResource writeResource = NULL;
CUgraphicsResource readResource = NULL;
CUarray writeArray, readArray;
CUdevice device;
CUcontext context;

// Which device to run on
unsigned int dev = 0;

// Default width, height, and iterations value
int width = 2048;
int height = 2048;
int itr = MAX_ITR;

// Error check variable
__device__ static unsigned int numErrors = 0;

//-----------------------FUNCTION PROTOTYPES------------------------//

void checkSync(int argc, char **argv);
int parseCmdLine(int argc, char **argv);
void printUsage(void);
void cleanup(int status);
void exitHandler(void);
void printStatus(int status);
void checkSyncOnCPU(void);
void checkSyncOnGPU(EGLDisplay dpy);

__global__ void verify_and_update_kernel(CUsurfObject write, CUsurfObject read,
                                         char expected, char newval, int width,
                                         int height);
extern "C" cudaError_t cudaGetValueMismatch();

//-----------------------FUNCTION DEFINITIONS------------------------//

int main(int argc, char *argv[]) {
#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  parseCmdLine(argc, argv);
  atexit(exitHandler);

  checkSync(argc, argv);
  return 0;
}

int parseCmdLine(int argc, char **argv) {
  int i;
  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-cpu") == 0) {
      useGpu = 0;
    }

    if (strcmp(argv[i], "-h") == 0) {
      printUsage();
      cleanup(SUCCESS);
    }

    if (strcmp(argv[i], "-width") == 0) {
      ++i;
      if (i == argc) {
        printf("width option must be followed by value\n");
        return FAILURE;
      }
      if (sscanf(argv[i], "%d", &width) != 1) {
        printf("Error: invalid width value\n");
        return FAILURE;
      }
    }

    if (strcmp(argv[i], "-height") == 0) {
      ++i;
      if (i == argc) {
        printf("height option must be followed by value\n");
        return FAILURE;
      }
      if (sscanf(argv[i], "%d", &height) != 1) {
        printf("Error: invalid height value\n");
        return FAILURE;
      }
    }
    if (strcmp(argv[i], "-itr") == 0) {
      ++i;
      if (i == argc) {
        printf("itr option must be followed by iteration value\n");
        return FAILURE;
      }
      if (sscanf(argv[i], "%d", &itr) != 1) {
        printf("Error: invalid iteration value\n");
        return FAILURE;
      }
    }
  }

  return SUCCESS;
}

void printUsage(void) {
  printf("Usage:\n");
  printf("\t-h\tPrint command line options\n");
  printf("\t-cpu\tSync on the CPU instead of the GPU\n");
  printf("\t-width w\tSet the width to w\n");
  printf("\t-height h\tSet the height to h\n");
  printf("\t-itr i\tSet number of iterations to i\n");
}

void checkSync(int argc, char **argv) {
  int x, y;
  int bufferSize = width * height * 4;
  unsigned char *pSurf_read = NULL, *pSurf_write = NULL;
  int integrated;

  CUresult status = CUDA_SUCCESS;

  // Init values for variables
  x = y = 0;

  if (CUDA_SUCCESS != (status = cuInit(0))) {
    printf("Failed to initialize CUDA\n");
  }
  device = findCudaDeviceDRV(argc, (const char **)argv);

  if (CUDA_SUCCESS != (status = cuCtxCreate(&context, 0, device))) {
    printf("failed to create CUDA context\n");
  }
  cuCtxPushCurrent(context);

  status =
      cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device);
  if (status != CUDA_SUCCESS) {
    printf("Failed to get device attribute CU_DEVICE_ATTRIBUTE_INTEGRATED\n");
    cleanup(FAILURE);
  }

  if (integrated != 1) {
    printf(
        "EGLSync_CUDAEvent_Interop does not support dGPU. Waiving sample.\n");
    cleanup(WAIVED);
  }

#if (defined(__arm__) || defined(__aarch64__)) && defined(__linux__)
  graphics_setup_window(0, 0, width, height, "EGLSync_CUDA_Interop");
#endif

  pSurf_read = (unsigned char *)malloc(bufferSize);
  pSurf_write = (unsigned char *)malloc(bufferSize);
  if (pSurf_read == NULL || pSurf_write == NULL) {
    printf("malloc failed\n");
    cleanup(FAILURE);
  }

  for (x = 0; x < width; x++) {
    for (y = 0; y < height; y++) {
      pSurf_read[(y * width + x) * 4] = 1;
      pSurf_read[(y * width + x) * 4 + 1] = 1;
      pSurf_read[(y * width + x) * 4 + 2] = 1;
      pSurf_read[(y * width + x) * 4 + 3] = 1;
      pSurf_write[(y * width + x) * 4] = 0;
      pSurf_write[(y * width + x) * 4 + 1] = 0;
      pSurf_write[(y * width + x) * 4 + 2] = 0;
      pSurf_write[(y * width + x) * 4 + 3] = 0;
    }
  }

  // NOP call to error-check the above glut calls
  GL_SAFE_CALL({});

  // Init texture
  GL_SAFE_CALL(glGenTextures(2, tex));

  GL_SAFE_CALL(glBindTexture(GL_TEXTURE_2D, tex[GL_READ]));
  GL_SAFE_CALL(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  GL_SAFE_CALL(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
  GL_SAFE_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                            GL_RGBA, GL_UNSIGNED_BYTE, pSurf_read));
  GL_SAFE_CALL(glBindTexture(GL_TEXTURE_2D, 0));

  GL_SAFE_CALL(glBindTexture(GL_TEXTURE_2D, tex[GL_WRITE]));
  GL_SAFE_CALL(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  GL_SAFE_CALL(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
  GL_SAFE_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                            GL_RGBA, GL_UNSIGNED_BYTE, pSurf_write));
  GL_SAFE_CALL(glBindTexture(GL_TEXTURE_2D, 0));

  glFinish();

  EGLDisplay eglDisplayHandle = eglGetCurrentDisplay();
  if (eglDisplayHandle == EGL_NO_DISPLAY) {
    printf("eglDisplayHandle failed \n");
    cleanup(FAILURE);
  } else {
    printf("eglDisplay Handle created \n");
  }

  if (!eglSetupExtensions()) {
    printf("SetupExtentions failed \n");
    cleanup(FAILURE);
  }

  EGLContext eglCtx = eglGetCurrentContext();
  if (eglCtx == EGL_NO_CONTEXT) {
    printf("Context1 create failed with error %d\n", eglGetError());
    cleanup(FAILURE);
  }

  // Create the EGL_Image
  EGLint eglImgAttrs[] = {EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE,
                          EGL_NONE};

  EGLImageKHR eglImage1 =
      eglCreateImageKHR(eglDisplayHandle, eglCtx, EGL_GL_TEXTURE_2D_KHR,
                        (EGLClientBuffer)(intptr_t)tex[GL_READ], eglImgAttrs);
  if (eglImage1 == EGL_NO_IMAGE_KHR) {
    printf("EGLImage create failed for read texture with error %d\n",
           eglGetError());
    cleanup(FAILURE);
  } else {
    printf("EGLImage1 created \n");
  }

  EGLImageKHR eglImage2 =
      eglCreateImageKHR(eglDisplayHandle, eglCtx, EGL_GL_TEXTURE_2D_KHR,
                        (EGLClientBuffer)(intptr_t)tex[GL_WRITE], eglImgAttrs);
  if (eglImage2 == EGL_NO_IMAGE_KHR) {
    printf("EGLImage create failed for write texture with error %d\n",
           eglGetError());
    cleanup(FAILURE);
  } else {
    printf("EGLImage2 created \n");
  }

  glFinish();

  status = cuGraphicsEGLRegisterImage(&writeResource, eglImage1,
                                      CU_GRAPHICS_REGISTER_FLAGS_NONE);
  if (status != CUDA_SUCCESS) {
    printf("cuGraphicsEGLRegisterImage failed with Texture 1\n");
    cleanup(FAILURE);
  } else {
    printf(
        "cuGraphicsEGLRegisterImage Passed, writeResource created with texture "
        "1\n");
  }

  status =
      cuGraphicsSubResourceGetMappedArray(&writeArray, writeResource, 0, 0);
  if (status != CUDA_SUCCESS) {
    printf(
        "cuGraphicsSubResourceGetMappedArray failed for writeResource with "
        "texture 1\n");
    cleanup(FAILURE);
  }

  status = cuGraphicsEGLRegisterImage(&readResource, eglImage2,
                                      CU_GRAPHICS_REGISTER_FLAGS_NONE);
  if (status != CUDA_SUCCESS) {
    printf(
        "cuGraphicsEGLRegisterImage failed for readResource with Texture 2\n");
    cleanup(FAILURE);
  } else {
    printf(
        "cuGraphicsEGLRegisterImage Passed, readResource created with texture "
        "2\n");
  }

  status = cuGraphicsSubResourceGetMappedArray(&readArray, readResource, 0, 0);
  if (status != CUDA_SUCCESS) {
    printf("cuGraphicsSubResourceGetMappedArray failed for texture 2\n");
    cleanup(FAILURE);
  }

  if (useGpu) {
    printf("Using GPU Sync path\n");
    checkSyncOnGPU(eglDisplayHandle);
  } else {
    printf("Using CPU Sync path\n");
    checkSyncOnCPU();
  }

  free(pSurf_read);
  free(pSurf_write);
  cleanup(SUCCESS);
}

void checkSyncOnCPU(void) {
  int z = 0;
  unsigned char expectedData, newData;
  CUresult status = CUDA_SUCCESS;
  CUDA_RESOURCE_DESC wdsc, rdsc;
  memset(&wdsc, 0, sizeof(wdsc));
  memset(&rdsc, 0, sizeof(rdsc));

  expectedData = 0;
  newData = 1;

  wdsc.resType = CU_RESOURCE_TYPE_ARRAY;
  wdsc.res.array.hArray = writeArray;
  CUsurfObject writeSurface;
  rdsc.resType = CU_RESOURCE_TYPE_ARRAY;
  rdsc.res.array.hArray = readArray;
  CUsurfObject readSurface;

  status = cuSurfObjectCreate(&writeSurface, &wdsc);
  if (status != CUDA_SUCCESS) {
    printf("Surface bounding failed with status %d\n", status);
    cleanup(FAILURE);
  }
  status = cuSurfObjectCreate(&readSurface, &rdsc);
  if (status != CUDA_SUCCESS) {
    printf("Surface bounding failed\n");
    cleanup(FAILURE);
  }

  for (z = 0; z < itr; z++) {
    // GL call to copy from read texture to write texture
    GL_SAFE_CALL(glCopyImageSubData(tex[GL_READ], GL_TEXTURE_2D, 0, 0, 0, 0,
                                    tex[GL_WRITE], GL_TEXTURE_2D, 0, 0, 0, 0,
                                    width, height, 1));

    glFinish();

    newData++;
    expectedData++;

    verify_and_update_kernel<<<(width * height) / 256, 256>>>(
        writeSurface, readSurface, expectedData, newData, width, height);

    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS) {
      printf("cuCtxSynchronize failed \n");
    }
  }

  cudaError_t err = cudaGetValueMismatch();
  if (err != cudaSuccess) {
    printf("Value mismatch seen when using CPU sync\n");
    cleanup(FAILURE);
  }

  // Clean up CUDA writeResource
  status = cuGraphicsUnregisterResource(writeResource);
  if (status != CUDA_SUCCESS) {
    printf("Failed to unregister %d", status);
    cleanup(FAILURE);
  } else {
    printf("Unregistered writeResource. \n");
  }

  // Clean up CUDA readResource
  status = cuGraphicsUnregisterResource(readResource);
  if (status != CUDA_SUCCESS) {
    printf("Failed to unregister %d", status);
    cleanup(FAILURE);
  } else {
    printf("Unregistered readResource. \n");
  }
}

/*
    Performs same function as checkSyncOnCPU
    Here instead of glFinish() and cuCtxSynchronize like in checkSyncOnCPU,
    we make use of EGLSync, CUDA Event and cuStreamWaitEvent, eglWaitSyncKHR to
   achieve the synchronization due to this CPU is not blocked for any
   synchronization needed between GL-EGL & CUDA operations all synchronizations
   happens on the GPU only.
*/
void checkSyncOnGPU(EGLDisplay dpy) {
  int z = 0;
  unsigned char expectedData, newData;
  cudaError_t err;
  CUresult status = CUDA_SUCCESS;
  CUstream stream;
  CUevent timingDisabledEvent;
  CUDA_RESOURCE_DESC wdsc, rdsc;
  memset(&wdsc, 0, sizeof(wdsc));
  memset(&rdsc, 0, sizeof(rdsc));

  expectedData = 0;
  newData = 1;

  wdsc.resType = CU_RESOURCE_TYPE_ARRAY;
  wdsc.res.array.hArray = writeArray;
  CUsurfObject writeSurface;
  rdsc.resType = CU_RESOURCE_TYPE_ARRAY;
  rdsc.res.array.hArray = readArray;
  CUsurfObject readSurface;

  status = cuSurfObjectCreate(&writeSurface, &wdsc);
  if (status != CUDA_SUCCESS) {
    printf("Surface bounding failed with status %d\n", status);
    cleanup(FAILURE);
  }
  status = cuSurfObjectCreate(&readSurface, &rdsc);
  if (status != CUDA_SUCCESS) {
    printf("Surface bounding failed\n");
    cleanup(FAILURE);
  }

  status = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
  if (status != CUDA_SUCCESS) {
    printf("Stream creation failed\n");
    cleanup(FAILURE);
  }

  // Creates timing disabled event which uses non-blocking synchronization
  status = cuEventCreate(&timingDisabledEvent, CU_EVENT_DISABLE_TIMING);
  if (status != CUDA_SUCCESS) {
    printf("Default event creation failed\n");
    cleanup(FAILURE);
  }

  /*
      1. We perform texture-to-texture copy in GLES which is async function
      2. Followed by creating EGLSync and a CUDA Event from that EGLSync object
      3. Using cuStreamWaitEvent() we wait in GPU for the GLES to finish texture
     copy.
      4. CUDA kernel verfiy_and_update_kernel verifies if the copied data by
     GLES is correct, and it updates the buffer with new values.
      5. This is followed by eglWaitSyncKHR() which waits for the cuda kernel to
     finish, so that in the next iteration GLES can perform the copying of the
     updated buffer to write texture,
  */
  for (z = 0; z < itr; z++) {
    // GL call to copy from read texture to write texture
    GL_SAFE_CALL(glCopyImageSubData(tex[GL_READ], GL_TEXTURE_2D, 0, 0, 0, 0,
                                    tex[GL_WRITE], GL_TEXTURE_2D, 0, 0, 0, 0,
                                    width, height, 1));

    EGLSyncKHR eglSyncForGL, eglSyncForCuda;
    EGLBoolean egl_status = EGL_TRUE;
    EGLAttribKHR eglattrib[] = {EGL_CUDA_EVENT_HANDLE_NV,
                                (EGLAttrib)timingDisabledEvent, EGL_NONE};

    CUevent cudaEGLSyncEvent;

    eglSyncForGL = eglCreateSyncKHR(dpy, EGL_SYNC_FENCE_KHR, NULL);

    if (eglSyncForGL == EGL_NO_SYNC_KHR) {
      printf(" EGL Sync creation failed\n");
      cleanup(FAILURE);
    }

    status = cuEventCreateFromEGLSync(&cudaEGLSyncEvent, eglSyncForGL,
                                      CU_EVENT_DEFAULT);
    if (status != CUDA_SUCCESS) {
      printf("CUDA event creation from EGLSync failed\n");
      cleanup(FAILURE);
    }

    // We wait from CUDA in GPU for GL-EGL operation completion
    status = cuStreamWaitEvent(stream, cudaEGLSyncEvent, 0);
    if (status != CUDA_SUCCESS) {
      printf("Stream wait for event created from EGLSync failed\n");
      cleanup(FAILURE);
    }

    egl_status = eglDestroySyncKHR(dpy, eglSyncForGL);
    if (egl_status != EGL_TRUE) {
      printf("EGL sync object destruction failed\n");
      cleanup(FAILURE);
    }

    newData++;
    expectedData++;

    // Verifies the values in readSurface which is copied by
    // glCopyImageSubData() And writes value of newData into writeSurface
    verify_and_update_kernel<<<(width * height) / 256, 256, 0, stream>>>(
        writeSurface, readSurface, expectedData, newData, width, height);

    status = cuEventDestroy(cudaEGLSyncEvent);
    if (status != CUDA_SUCCESS) {
      printf("Event Destroy failed\n");
      cleanup(FAILURE);
    }

    status = cuEventRecord(timingDisabledEvent, stream);
    if (status != CUDA_SUCCESS) {
      printf("Event Record failed\n");
      cleanup(FAILURE);
    }

    // creating an EGL sync object linked to a CUDA event object
    eglSyncForCuda = eglCreateSync64KHR(dpy, EGL_SYNC_CUDA_EVENT_NV, eglattrib);

    // We wait from EGL for CUDA operation completion
    egl_status = eglWaitSyncKHR(dpy, eglSyncForCuda, 0);
    if (egl_status != EGL_TRUE) {
      printf("eglWaitSyncKHR failed\n");
      cleanup(FAILURE);
    }
    egl_status = eglDestroySyncKHR(dpy, eglSyncForCuda);
    if (egl_status != EGL_TRUE) {
      printf("EGL sync object destruction failed\n");
      cleanup(FAILURE);
    }
  }

  err = cudaGetValueMismatch();
  if (err != cudaSuccess) {
    printf("Value mismatch seen when using GPU sync\n");
    cleanup(FAILURE);
  }

  // Clean up CUDA writeResource
  status = cuGraphicsUnregisterResource(writeResource);
  if (status != CUDA_SUCCESS) {
    printf("Failed to unregister %d", status);
    cleanup(FAILURE);
  } else {
    printf("Unregistered writeResource. \n");
  }

  // Clean up CUDA readResource
  status = cuGraphicsUnregisterResource(readResource);
  if (status != CUDA_SUCCESS) {
    printf("Failed to unregister %d", status);
    cleanup(FAILURE);
  } else {
    printf("Unregistered readResource. \n");
  }
}

// Verifies the values in readSurface whether they are expected ones
// And writes value of newData into writeSurface
__global__ void verify_and_update_kernel(CUsurfObject write, CUsurfObject read,
                                         char expected, char newval, int width,
                                         int height) {
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < width && y < height) {
    uchar4 check;
    surf2Dread(&check, read, x * 4, y);
    if (check.x != expected || check.y != expected || check.z != expected ||
        check.w != expected) {
      printf(
          "Mismatch found in values read[0]= %u read[1]= %u read[2]= %u "
          "read[3]= %u expected is %u\n",
          check.x, check.y, check.z, check.w, expected);
      numErrors++;
      return;
    }
    uchar4 data = make_uchar4(newval, newval, newval, newval);
    surf2Dwrite(data, write, x * 4, y);
  }
}

__global__ void getNumErrors(int *numErr) { *numErr = numErrors; }

extern "C" cudaError_t cudaGetValueMismatch() {
  int numErr_h;
  int *numErr_d = NULL;
  cudaError_t err = cudaSuccess;

  err = cudaMalloc(&numErr_d, sizeof(int));
  if (err != cudaSuccess) {
    printf("Cuda Main: cudaMemcpy failed with %s\n", cudaGetErrorString(err));
    cudaFree(numErr_d);
    return err;
  }

  getNumErrors<<<1, 1>>>(numErr_d);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Cuda Main: cudaDeviceSynchronize failed with %s\n",
           cudaGetErrorString(err));
  }
  err = cudaMemcpy(&numErr_h, numErr_d, sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("Cuda Main: cudaMemcpy failed with %s\n", cudaGetErrorString(err));
    cudaFree(numErr_d);
    return err;
  }
  err = cudaFree(numErr_d);
  if (err != cudaSuccess) {
    printf("Cuda Main: cudaFree failed with %s\n", cudaGetErrorString(err));
    return err;
  }
  if (numErr_h > 0) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

// Clean up state and exit. If status is SUCCESS, regression success is printed
// to stdout. This will happen if the glut timer is triggered. If status is
// anything else, the regression failure message is printed.
void cleanup(int status) {
  GLenum glErr = GL_NO_ERROR;
  cudaError cudaErr = cudaSuccess;
  int exitStatus = status;

  // Clean up GL
  if (*tex) {
    GL_SAFE_CALL_NO_CLEANUP(glDeleteTextures(2, tex), glErr);
  }

  // Print test status and exit
  if (glErr != GL_NO_ERROR || cudaErr != cudaSuccess) exitStatus = FAILURE;

  printStatus(exitStatus);

  cleanExit = 1;

  graphics_close_window();

  if (exitStatus == FAILURE) exit(EXIT_FAILURE);

  if (exitStatus == WAIVED) exit(EXIT_WAIVED);

  exit(0);
}

void exitHandler(void) {
  if (!cleanExit) {
    printf("&&&& EGLSync_CUDAEvent_Interop unexpected failure \n");
    printStatus(FAILURE);
  }
}

// Print test success or fail for regression testing
void printStatus(int status) {
  switch (status) {
    case SUCCESS:
      printf("&&&& EGLSync_CUDAEvent_Interop PASSED\n");
      break;
    case WAIVED:
      printf("&&&& EGLSync_CUDAEvent_Interop WAIVED\n");
      break;
    default:
      printf("&&&& EGLSync_CUDAEvent_Interop FAILED\n");
      break;
  }
  fflush(stdout);
}
