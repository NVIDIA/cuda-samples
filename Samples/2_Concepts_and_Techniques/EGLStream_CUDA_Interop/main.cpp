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
// DESCRIPTION:   Simple EGL stream sample app
//
//

//#define EGL_EGLEXT_PROTOTYPES

#include "cudaEGL.h"
#include "cuda_consumer.h"
#include "cuda_producer.h"
#include "eglstrm_common.h"

/* ------  globals ---------*/

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif

#define NUM_TRAILS 4

bool signal_stop = 0;

static void sig_handler(int sig) {
  signal_stop = 1;
  printf("Signal: %d\n", sig);
}

int main(int argc, char **argv) {
  TestArgs args;
  CUresult curesult = CUDA_SUCCESS;
  unsigned int i, j;
  EGLint streamState = 0;

  test_cuda_consumer_s cudaConsumer;
  test_cuda_producer_s cudaProducer;

  memset(&cudaProducer, 0, sizeof(test_cuda_producer_s));
  memset(&cudaConsumer, 0, sizeof(test_cuda_consumer_s));

  // Hook up Ctrl-C handler
  signal(SIGINT, sig_handler);
  if (!eglSetupExtensions()) {
    printf("SetupExtentions failed \n");
    curesult = CUDA_ERROR_UNKNOWN;
    goto done;
  }

  checkCudaErrors(cuInit(0));

  int count;

  checkCudaErrors(cuDeviceGetCount(&count));
  printf("Found %d cuda devices\n", count);

  CUdevice devId;

  if (!EGLStreamInit(&devId)) {
    printf("EGLStream Init failed.\n");
    curesult = CUDA_ERROR_UNKNOWN;
    goto done;
  }
  curesult = cudaDeviceCreateProducer(&cudaProducer, devId);
  if (curesult != CUDA_SUCCESS) {
    goto done;
  }
  curesult = cudaDeviceCreateConsumer(&cudaConsumer, devId);
  if (curesult != CUDA_SUCCESS) {
    goto done;
  }
  checkCudaErrors(cuCtxPushCurrent(cudaConsumer.context));
  if (CUDA_SUCCESS != (curesult = cuEGLStreamConsumerConnect(
                           &(cudaConsumer.cudaConn), eglStream))) {
    printf("FAILED Connect CUDA consumer  with error %d\n", curesult);
    goto done;
  } else {
    printf("Connected CUDA consumer, CudaConsumer %p\n", cudaConsumer.cudaConn);
  }
  checkCudaErrors(cuCtxPopCurrent(&cudaConsumer.context));

  checkCudaErrors(cuCtxPushCurrent(cudaProducer.context));
  if (CUDA_SUCCESS ==
      (curesult = cuEGLStreamProducerConnect(&(cudaProducer.cudaConn),
                                             eglStream, WIDTH, HEIGHT))) {
    printf("Connect CUDA producer Done, CudaProducer %p\n",
           cudaProducer.cudaConn);
  } else {
    printf("Connect CUDA producer FAILED with error %d\n", curesult);
    goto done;
  }
  checkCudaErrors(cuCtxPopCurrent(&cudaProducer.context));

  // Initialize producer
  for (i = 0; i < NUM_TRAILS; i++) {
    if (streamState != EGL_STREAM_STATE_CONNECTING_KHR) {
      if (!eglQueryStreamKHR(g_display, eglStream, EGL_STREAM_STATE_KHR,
                             &streamState)) {
        printf("main: eglQueryStreamKHR EGL_STREAM_STATE_KHR failed\n");
        curesult = CUDA_ERROR_UNKNOWN;
        goto done;
      }
    }
    args.inputWidth = WIDTH;
    args.inputHeight = HEIGHT;
    if (i % 2 != 0) {
      args.isARGB = 1;
      args.infile1 = sdkFindFilePath("cuda_f_1.yuv", argv[0]);
      args.infile2 = sdkFindFilePath("cuda_f_2.yuv", argv[0]);
    } else {
      args.isARGB = 0;
      args.infile1 = sdkFindFilePath("cuda_yuv_f_1.yuv", argv[0]);
      args.infile2 = sdkFindFilePath("cuda_yuv_f_2.yuv", argv[0]);
    }
    if ((i % 4) < 2) {
      args.pitchLinearOutput = 1;
    } else {
      args.pitchLinearOutput = 0;
    }

    checkCudaErrors(cuCtxPushCurrent(cudaProducer.context));
    cudaProducerInit(&cudaProducer, g_display, eglStream, &args);
    checkCudaErrors(cuCtxPopCurrent(&cudaProducer.context));

    checkCudaErrors(cuCtxPushCurrent(cudaConsumer.context));
    cuda_consumer_init(&cudaConsumer, &args);
    checkCudaErrors(cuCtxPopCurrent(&cudaConsumer.context));

    printf("main - Cuda Producer and Consumer Initialized.\n");

    for (j = 0; j < 2; j++) {
      printf("Running for %s frame and %s input\n",
             args.isARGB ? "ARGB" : "YUV",
             args.pitchLinearOutput ? "Pitchlinear" : "BlockLinear");
      if (j == 0) {
        checkCudaErrors(cuCtxPushCurrent(cudaProducer.context));
        curesult = cudaProducerTest(&cudaProducer, cudaProducer.fileName1);
        if (curesult != CUDA_SUCCESS) {
          printf("Cuda Producer Test failed for frame = %d\n", j + 1);
          goto done;
        }
        checkCudaErrors(cuCtxPopCurrent(&cudaProducer.context));
        checkCudaErrors(cuCtxPushCurrent(cudaConsumer.context));
        curesult = cudaConsumerTest(&cudaConsumer, cudaConsumer.outFile1);
        if (curesult != CUDA_SUCCESS) {
          printf("Cuda Consumer Test failed for frame = %d\n", j + 1);
          goto done;
        }
        checkCudaErrors(cuCtxPopCurrent(&cudaConsumer.context));
      } else {
        checkCudaErrors(cuCtxPushCurrent(cudaProducer.context));
        curesult = cudaProducerTest(&cudaProducer, cudaProducer.fileName2);
        if (curesult != CUDA_SUCCESS) {
          printf("Cuda Producer Test failed for frame = %d\n", j + 1);
          goto done;
        }

        checkCudaErrors(cuCtxPopCurrent(&cudaProducer.context));
        checkCudaErrors(cuCtxPushCurrent(cudaConsumer.context));
        curesult = cudaConsumerTest(&cudaConsumer, cudaConsumer.outFile2);
        if (curesult != CUDA_SUCCESS) {
          printf("Cuda Consumer Test failed for frame = %d\n", j + 1);
          goto done;
        }
        checkCudaErrors(cuCtxPopCurrent(&cudaConsumer.context));
      }
    }
  }

  checkCudaErrors(cuCtxPushCurrent(cudaProducer.context));
  if (CUDA_SUCCESS != (curesult = cudaProducerDeinit(&cudaProducer))) {
    printf("Producer Disconnect FAILED. \n");
    goto done;
  }
  checkCudaErrors(cuCtxPopCurrent(&cudaProducer.context));

  if (!eglQueryStreamKHR(g_display, eglStream, EGL_STREAM_STATE_KHR,
                         &streamState)) {
    printf("Cuda consumer, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed\n");
    curesult = CUDA_ERROR_UNKNOWN;
    goto done;
  }
  if (streamState != EGL_STREAM_STATE_DISCONNECTED_KHR) {
    if (CUDA_SUCCESS != (curesult = cuda_consumer_deinit(&cudaConsumer))) {
      printf("Consumer Disconnect FAILED.\n");
      goto done;
    }
  }
  printf("Producer and Consumer Disconnected \n");

done:
  if (!eglQueryStreamKHR(g_display, eglStream, EGL_STREAM_STATE_KHR,
                         &streamState)) {
    printf("Cuda consumer, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed\n");
    curesult = CUDA_ERROR_UNKNOWN;
  }
  if (streamState != EGL_STREAM_STATE_DISCONNECTED_KHR) {
    EGLStreamFini();
  }

  if (curesult == CUDA_SUCCESS) {
    printf("&&&& EGLStream interop test PASSED\n");
  } else {
    printf("&&&& EGLStream interop test FAILED\n");
  }
  return 0;
}
