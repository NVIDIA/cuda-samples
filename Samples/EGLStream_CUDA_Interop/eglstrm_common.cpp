/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

//
// DESCRIPTION:   Common egl stream functions
//

#include "eglstrm_common.h"

EGLStreamKHR eglStream;
EGLDisplay g_display;
EGLAttrib cudaIndex;

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

int EGLStreamInit(int *cuda_device) {
  static const EGLint streamAttrMailboxMode[] = {EGL_SUPPORT_REUSE_NV,
                                                 EGL_FALSE, EGL_NONE};
  EGLBoolean eglStatus;
#define MAX_EGL_DEVICES 4
  EGLint numDevices = 0;
  EGLDeviceEXT devices[MAX_EGL_DEVICES];
  eglStatus = eglQueryDevicesEXT(MAX_EGL_DEVICES, devices, &numDevices);
  if (eglStatus != EGL_TRUE) {
    printf("Error querying EGL devices\n");
    exit(EXIT_FAILURE);
  }

  if (numDevices == 0) {
    printf("No EGL devices found.. Waiving\n");
    eglStatus = EGL_FALSE;
    exit(EXIT_WAIVED);
  }

  int egl_device_id = 0;
  for (egl_device_id = 0; egl_device_id < numDevices; egl_device_id++) {
    eglStatus = eglQueryDeviceAttribEXT(devices[egl_device_id],
                                        EGL_CUDA_DEVICE_NV, &cudaIndex);
    if (eglStatus == EGL_TRUE) {
      *cuda_device = cudaIndex;  // We select first EGL-CUDA Capable device.
      printf("Found EGL-CUDA Capable device with CUDA Device id = %d\n",
             (int)cudaIndex);
      break;
    }
  }

  if (egl_device_id >= numDevices) {
    printf("No CUDA Capable EGL Device found.. Waiving execution\n");
    exit(EXIT_WAIVED);
  }

  g_display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                       (void *)devices[egl_device_id], NULL);
  if (g_display == EGL_NO_DISPLAY) {
    printf("Could not get EGL display from device. \n");
    eglStatus = EGL_FALSE;
    exit(EXIT_FAILURE);
  }

  eglStatus = eglInitialize(g_display, 0, 0);
  if (!eglStatus) {
    printf("EGL failed to initialize. \n");
    eglStatus = EGL_FALSE;
    exit(EXIT_FAILURE);
  }

  eglStream = eglCreateStreamKHR(g_display, streamAttrMailboxMode);
  if (eglStream == EGL_NO_STREAM_KHR) {
    printf("Could not create EGL stream.\n");
    eglStatus = EGL_FALSE;
    exit(EXIT_FAILURE);
  }

  printf("Created EGLStream %p\n", eglStream);

  // Set stream attribute
  if (!eglStreamAttribKHR(g_display, eglStream, EGL_CONSUMER_LATENCY_USEC_KHR,
                          16000)) {
    printf(
        "Consumer: eglStreamAttribKHR EGL_CONSUMER_LATENCY_USEC_KHR failed\n");
    return 0;
  }
  if (!eglStreamAttribKHR(g_display, eglStream,
                          EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, 16000)) {
    printf(
        "Consumer: eglStreamAttribKHR EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR "
        "failed\n");
    return 0;
  }
  printf("EGLStream initialized\n");
  return 1;
}

void EGLStreamFini(void) { eglDestroyStreamKHR(g_display, eglStream); }
#endif
