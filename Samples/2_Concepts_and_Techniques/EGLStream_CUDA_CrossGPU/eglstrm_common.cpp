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

//
// DESCRIPTION:   Common egl stream functions
//

#include "eglstrm_common.h"

EGLStreamKHR g_producerEglStream = EGL_NO_STREAM_KHR;
EGLStreamKHR g_consumerEglStream = EGL_NO_STREAM_KHR;
EGLDisplay g_producerEglDisplay = EGL_NO_DISPLAY;
EGLDisplay g_consumerEglDisplay = EGL_NO_DISPLAY;
int cudaDevIndexProd = -1;
int cudaDevIndexCons = -1;

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_DECL)
typedef void (*extlst_fnptr_t)(void);
static struct {
  extlst_fnptr_t *fnptr;
  char const *name;
  bool is_dgpu;  // This function is need only for dgpu case
} extensionList[] = {EXTENSION_LIST(EXTLST_ENTRY)};

int eglSetupExtensions(bool isCrossDevice) {
  unsigned int i;

  for (i = 0; i < (sizeof(extensionList) / sizeof(*extensionList)); i++) {
    // load the dgpu function only if we are running cross device test
    if ((!extensionList[i].is_dgpu) ||
        (extensionList[i].is_dgpu == isCrossDevice)) {
      *extensionList[i].fnptr = eglGetProcAddress(extensionList[i].name);
      if (*extensionList[i].fnptr == NULL) {
        printf("Couldn't get address of %s()\n", extensionList[i].name);
        return 0;
      }
    }
  }

  return 1;
}

int EGLStreamInit(bool isCrossDevice, int isConsumer,
                  EGLNativeFileDescriptorKHR fileDesc) {
  static const EGLint streamAttrFIFOMode[] = {
      EGL_STREAM_FIFO_LENGTH_KHR, 5, EGL_SUPPORT_REUSE_NV, EGL_FALSE, EGL_NONE};
  EGLDisplay eglDisplay[2] = {0};
  EGLStreamKHR eglStream[2] = {0};
  EGLBoolean eglStatus;

#define MAX_EGL_DEVICES 4

  EGLDeviceEXT devices[MAX_EGL_DEVICES];
  EGLint numDevices = 0;

  eglStatus = eglQueryDevicesEXT(MAX_EGL_DEVICES, devices, &numDevices);
  if (eglStatus != EGL_TRUE) {
    printf("Error querying EGL devices\n");
    goto Done;
  }

  if (numDevices == 0) {
    printf("No EGL devices found\n");
    eglStatus = EGL_FALSE;
    goto Done;
  }

  // If cross device, create discrete GPU stream first and then create the
  // integrated GPU stream to connect to it via fd. The other way round fails
  // in producer connect.
  //
  // TODO: Find out if this EGL behavior is by design.
  if (isConsumer) {
    int egl_device_id = 0;
    for (egl_device_id = 0; egl_device_id < numDevices; egl_device_id++) {
      EGLAttrib cuda_device;
      eglStatus = eglQueryDeviceAttribEXT(devices[egl_device_id],
                                          EGL_CUDA_DEVICE_NV, &cuda_device);
      if (eglStatus == EGL_TRUE) {
        cudaDevIndexCons = cuda_device;  // We select first EGL-CUDA Capable
                                         // device for consumer.
        printf(
            "Found EGL-CUDA Capable device with CUDA Device id = %d out of "
            "egl_device_id = %d\n",
            (int)cudaDevIndexCons, egl_device_id);
        break;
      }
    }

    if (egl_device_id >= numDevices) {
      printf("No CUDA Capable EGL Device found.. Waiving execution\n");
      goto Done;
    }

    g_consumerEglDisplay = eglGetPlatformDisplayEXT(
        EGL_PLATFORM_DEVICE_EXT, (void *)devices[egl_device_id], NULL);
    if (g_consumerEglDisplay == EGL_NO_DISPLAY) {
      printf("Could not get EGL display from device. \n");
      eglStatus = EGL_FALSE;
      goto Done;
    }

    eglStatus = eglInitialize(g_consumerEglDisplay, 0, 0);
    if (!eglStatus) {
      printf("EGL failed to initialize. \n");
      eglStatus = EGL_FALSE;
      goto Done;
    }

    g_consumerEglStream =
        eglCreateStreamKHR(g_consumerEglDisplay, streamAttrFIFOMode);
    if (g_consumerEglStream == EGL_NO_STREAM_KHR) {
      printf("Could not create EGL stream.\n");
      eglStatus = EGL_FALSE;
      goto Done;
    }

    eglStatus = eglStreamAttribKHR(g_consumerEglDisplay, g_consumerEglStream,
                                   EGL_CONSUMER_LATENCY_USEC_KHR, 16000);
    if (eglStatus != EGL_TRUE) {
      printf("eglStreamAttribKHR EGL_CONSUMER_LATENCY_USEC_KHR failed\n");
      goto Done;
    }

    eglStatus =
        eglStreamAttribKHR(g_consumerEglDisplay, g_consumerEglStream,
                           EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, 16000);
    if (eglStatus != EGL_TRUE) {
      printf(
          "eglStreamAttribKHR EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR failed\n");
      goto Done;
    }
  }

  if (!isConsumer) {  // Producer

    if (fileDesc == EGL_NO_FILE_DESCRIPTOR_KHR) {
      printf("Cuda Producer received bad file descriptor\n");
      eglStatus = EGL_FALSE;
      goto Done;
    }

    int egl_device_id = 0;
    int egl_cuda_devices = 0;
    for (egl_device_id = 0; egl_device_id < numDevices; egl_device_id++) {
      EGLAttrib cuda_device = -1;
      eglStatus = eglQueryDeviceAttribEXT(devices[egl_device_id],
                                          EGL_CUDA_DEVICE_NV, &cuda_device);
      if (eglStatus == EGL_TRUE) {
        egl_cuda_devices++;
        if (isCrossDevice && (egl_cuda_devices > 1)) {
          // We select second EGL-CUDA Capable device for producer.
          cudaDevIndexProd = (int)cuda_device;
          printf(
              "Found EGL-CUDA Capable device with CUDA Device id = %d "
              "egl_device_id = %d \n",
              (int)cudaDevIndexProd, egl_device_id);
          break;
        }
        if (!isCrossDevice) {
          // We select first EGL-CUDA Capable device for producer same as
          // consumer.
          cudaDevIndexProd = (int)cuda_device;
          printf(
              "Found EGL-CUDA Capable device with CUDA Device id = %d "
              "egl_device_id = %d \n",
              (int)cudaDevIndexProd, egl_device_id);
          break;
        }
      }
    }

    if (egl_device_id >= numDevices) {
      printf("No CUDA Capable EGL Device found.. Waiving execution\n");
      goto Done;
    }

    g_producerEglDisplay = eglGetPlatformDisplayEXT(
        EGL_PLATFORM_DEVICE_EXT, (void *)devices[egl_device_id], NULL);
    if (g_producerEglDisplay == EGL_NO_DISPLAY) {
      printf("Could not get EGL display from device. \n");
      eglStatus = EGL_FALSE;
      goto Done;
    }

    eglStatus = eglInitialize(g_producerEglDisplay, 0, 0);
    if (!eglStatus) {
      printf("EGL failed to initialize. \n");
      eglStatus = EGL_FALSE;
      goto Done;
    }

    g_producerEglStream =
        eglCreateStreamFromFileDescriptorKHR(g_producerEglDisplay, fileDesc);
    close(fileDesc);

    if (g_producerEglStream == EGL_NO_STREAM_KHR) {
      printf("CUDA Producer Could not create EGL stream.\n");
      eglStatus = EGL_FALSE;
      goto Done;
    } else {
      printf("Producer created EGLStream for the GPU.\n");
    }
  }

Done:
  return eglStatus == EGL_TRUE ? 1 : 0;
}

void EGLStreamFini(void) {
  if (g_producerEglStream != EGL_NO_STREAM_KHR) {
    eglDestroyStreamKHR(g_producerEglDisplay, g_producerEglStream);
  }
  if (g_consumerEglStream != g_producerEglStream) {
    if (g_consumerEglStream != EGL_NO_STREAM_KHR) {
      eglDestroyStreamKHR(g_consumerEglDisplay, g_consumerEglStream);
    }
  }
}

int UnixSocketConnect(const char *socket_name) {
  int sock_fd = -1;
  struct sockaddr_un sock_addr;
  int wait_loop = 0;

  sock_fd = socket(PF_UNIX, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    printf("%s: socket create failed.\n", __func__);
    return -1;
  }

  if (verbose) printf("%s: send_fd: sock_fd: %d\n", __func__, sock_fd);

  memset(&sock_addr, 0, sizeof(struct sockaddr_un));
  sock_addr.sun_family = AF_UNIX;
  strncpy(sock_addr.sun_path, socket_name, sizeof(sock_addr.sun_path) - 1);

  while (connect(sock_fd, (const struct sockaddr *)&sock_addr,
                 sizeof(struct sockaddr_un))) {
    if (wait_loop < 60) {
      if (!wait_loop)
        printf("Waiting for EGL stream producer ");
      else
        printf(".");
      fflush(stdout);
      sleep(1);
      wait_loop++;
    } else {
      printf("\n%s: Waiting timed out\n", __func__);
      return -1;
    }
  }
  if (wait_loop) printf("\n");

  if (verbose) printf("%s: Wait is done\n", __func__);

  return sock_fd;
}

/* Send <fd_to_send> (a file descriptor) to another process */
/* over a unix domain socket named <socket_name>.           */
/* <socket_name> can be any nonexistant filename.           */
int EGLStreamSendfd(int send_fd, int fd_to_send) {
  struct msghdr msg;
  struct iovec iov[1];
  char ctrl_buf[CMSG_SPACE(sizeof(int))];
  struct cmsghdr *cmsg = NULL;
  void *data;
  int res;
  memset(&msg, 0, sizeof(msg));

  iov[0].iov_len = 1;             // must send at least 1 byte
  iov[0].iov_base = (void *)"x";  // any byte value (value ignored)
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  memset(ctrl_buf, 0, sizeof(ctrl_buf));
  msg.msg_control = ctrl_buf;
  msg.msg_controllen = sizeof(ctrl_buf);

  cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  data = CMSG_DATA(cmsg);
  *(int *)data = fd_to_send;

  msg.msg_controllen = cmsg->cmsg_len;

  res = sendmsg(send_fd, &msg, 0);
  if (res <= 0) {
    printf("%s: sendmsg failed", __func__);
    return -1;
  }

  return 0;
}

/* Listen on a unix domain socket named <socket_name>.     */
/* Connect to it and return connect_fd                     */
int UnixSocketCreate(const char *socket_name) {
  int listen_fd;
  struct sockaddr_un sock_addr;
  int connect_fd;
  struct sockaddr_un connect_addr;
  socklen_t connect_addr_len = 0;

  listen_fd = socket(PF_UNIX, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    printf("%s: socket create failed", __func__);
    return -1;
  }

  if (verbose) printf("%s: listen_fd: %d\n", __func__, listen_fd);

  unlink(socket_name);

  memset(&sock_addr, 0, sizeof(struct sockaddr_un));
  sock_addr.sun_family = AF_UNIX;
  strncpy(sock_addr.sun_path, socket_name, sizeof(sock_addr.sun_path) - 1);

  if (bind(listen_fd, (const struct sockaddr *)&sock_addr,
           sizeof(struct sockaddr_un))) {
    printf("i%s: bind error", __func__);
    return -1;
  }

  if (listen(listen_fd, 1)) {
    printf("%s: listen error", __func__);
    return -1;
  }

  connect_fd =
      accept(listen_fd, (struct sockaddr *)&connect_addr, &connect_addr_len);

  if (verbose) printf("%s: connect_fd: %d\n", __func__, connect_fd);

  close(listen_fd);
  unlink(socket_name);
  if (connect_fd < 0) {
    printf("%s: accept failed\n", __func__);
    return -1;
  }

  return connect_fd;
}

/* receive a file descriptor from another process.         */
/* Returns the file descriptor.  Note: the integer value   */
/* of the file descriptor may be different from the        */
/* integer value in the other process, but the file        */
/* descriptors in each process will refer to the same file */
/* object in the kernel.                                   */
int EGLStreamReceivefd(int connect_fd) {
  struct msghdr msg;
  struct iovec iov[1];
  char msg_buf[1];
  char ctrl_buf[CMSG_SPACE(sizeof(int))];
  struct cmsghdr *cmsg;
  void *data;
  int recvfd;

  memset(&msg, 0, sizeof(msg));

  iov[0].iov_base = msg_buf;
  iov[0].iov_len = sizeof(msg_buf);
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  msg.msg_control = ctrl_buf;
  msg.msg_controllen = sizeof(ctrl_buf);

  if (recvmsg(connect_fd, &msg, 0) <= 0) {
    printf("%s: recvmsg failed", __func__);
    return -1;
  }

  cmsg = CMSG_FIRSTHDR(&msg);
  if (!cmsg) {
    printf("%s: NULL message header\n", __func__);
    return -1;
  }
  if (cmsg->cmsg_level != SOL_SOCKET) {
    printf("%s: Message level is not SOL_SOCKET\n", __func__);
    return -1;
  }
  if (cmsg->cmsg_type != SCM_RIGHTS) {
    printf("%s: Message type is not SCM_RIGHTS\n", __func__);
    return -1;
  }

  data = CMSG_DATA(cmsg);
  recvfd = *(int *)data;

  return recvfd;
}

#endif
