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
// DESCRIPTION:   Simple cuda EGL stream producer app
//

#include "cuda_producer.h"
#include <helper_cuda_drvapi.h>
#include "cudaEGL.h"
#include "eglstrm_common.h"

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif

static CUresult cudaProducerReadYUVFrame(FILE *file, unsigned int frameNum,
                                         unsigned int width,
                                         unsigned int height,
                                         unsigned char *pBuff) {
  int bOrderUV = 0;
  unsigned char *pYBuff, *pUBuff, *pVBuff, *pChroma;
  unsigned int frameSize = (width * height * 3) / 2;
  CUresult ret = CUDA_SUCCESS;
  unsigned int i;

  if (!pBuff || !file) return CUDA_ERROR_FILE_NOT_FOUND;

  pYBuff = pBuff;

  // YVU order in the buffer
  pVBuff = pYBuff + width * height;
  pUBuff = pVBuff + width * height / 4;

  if (fseek(file, frameNum * frameSize, SEEK_SET)) {
    printf("ReadYUVFrame: Error seeking file: %p\n", file);
    ret = CUDA_ERROR_NOT_PERMITTED;
    goto done;
  }
  // read Y U V separately
  for (i = 0; i < height; i++) {
    if (fread(pYBuff, width, 1, file) != 1) {
      printf("ReadYUVFrame: Error reading file: %p\n", file);
      ret = CUDA_ERROR_NOT_PERMITTED;
      goto done;
    }
    pYBuff += width;
  }

  pChroma = bOrderUV ? pUBuff : pVBuff;
  for (i = 0; i < height / 2; i++) {
    if (fread(pChroma, width / 2, 1, file) != 1) {
      printf("ReadYUVFrame: Error reading file: %p\n", file);
      ret = CUDA_ERROR_NOT_PERMITTED;
      goto done;
    }
    pChroma += width / 2;
  }

  pChroma = bOrderUV ? pVBuff : pUBuff;
  for (i = 0; i < height / 2; i++) {
    if (fread(pChroma, width / 2, 1, file) != 1) {
      printf("ReadYUVFrame: Error reading file: %p\n", file);
      ret = CUDA_ERROR_NOT_PERMITTED;
      goto done;
    }
    pChroma += width / 2;
  }
done:
  return ret;
}

static CUresult cudaProducerReadARGBFrame(FILE *file, unsigned int frameNum,
                                          unsigned int width,
                                          unsigned int height,
                                          unsigned char *pBuff) {
  unsigned int frameSize = width * height * 4;
  CUresult ret = CUDA_SUCCESS;

  if (!pBuff || !file) return CUDA_ERROR_FILE_NOT_FOUND;

  if (fseek(file, frameNum * frameSize, SEEK_SET)) {
    printf("ReadYUVFrame: Error seeking file: %p\n", file);
    ret = CUDA_ERROR_NOT_PERMITTED;
    goto done;
  }

  // read ARGB data
  if (fread(pBuff, frameSize, 1, file) != 1) {
    if (feof(file))
      printf("ReadARGBFrame: file read to the end\n");
    else
      printf("ReadARGBFrame: Error reading file: %p\n", file);
    ret = CUDA_ERROR_NOT_PERMITTED;
    goto done;
  }
done:
  return ret;
}

CUresult cudaProducerTest(test_cuda_producer_s *cudaProducer, char *file) {
  int framenum = 0;
  CUarray cudaArr[3] = {0};
  CUdeviceptr cudaPtr[3] = {0, 0, 0};
  unsigned int bufferSize;
  CUresult cuStatus = CUDA_SUCCESS;
  unsigned int i, surfNum, uvOffset[3] = {0};
  unsigned int copyWidthInBytes[3] = {0, 0, 0}, copyHeight[3] = {0, 0, 0};
  CUeglColorFormat eglColorFormat;
  FILE *file_p;
  CUeglFrame cudaEgl;
  CUcontext oldContext;

  file_p = fopen(file, "rb");
  if (!file_p) {
    printf("CudaProducer: Error opening file: %s\n", file);
    goto done;
  }

  if (cudaProducer->pitchLinearOutput) {
    if (cudaProducer->isARGB) {
      cudaPtr[0] = cudaProducer->cudaPtrARGB[0];
    } else {  // YUV case
      for (i = 0; i < 3; i++) {
        if (i == 0) {
          bufferSize = cudaProducer->width * cudaProducer->height;
        } else {
          bufferSize = cudaProducer->width * cudaProducer->height / 4;
        }

        cudaPtr[i] = cudaProducer->cudaPtrYUV[i];
      }
    }
  } else {
    if (cudaProducer->isARGB) {
      cudaArr[0] = cudaProducer->cudaArrARGB[0];
    } else {
      for (i = 0; i < 3; i++) {
        cudaArr[i] = cudaProducer->cudaArrYUV[i];
      }
    }
  }
  uvOffset[0] = 0;
  if (cudaProducer->isARGB) {
    if (CUDA_SUCCESS !=
        cudaProducerReadARGBFrame(file_p, framenum, cudaProducer->width,
                                  cudaProducer->height, cudaProducer->pBuff)) {
      printf("cuda producer, read ARGB frame failed\n");
      goto done;
    }
    copyWidthInBytes[0] = cudaProducer->width * 4;
    copyHeight[0] = cudaProducer->height;
    surfNum = 1;
    eglColorFormat = CU_EGL_COLOR_FORMAT_ARGB;
  } else {
    if (CUDA_SUCCESS !=
        cudaProducerReadYUVFrame(file_p, framenum, cudaProducer->width,
                                 cudaProducer->height, cudaProducer->pBuff)) {
      printf("cuda producer, reading YUV frame failed\n");
      goto done;
    }
    surfNum = 3;
    eglColorFormat = CU_EGL_COLOR_FORMAT_YUV420_PLANAR;
    copyWidthInBytes[0] = cudaProducer->width;
    copyHeight[0] = cudaProducer->height;
    copyWidthInBytes[1] = cudaProducer->width / 2;
    copyHeight[1] = cudaProducer->height / 2;
    copyWidthInBytes[2] = cudaProducer->width / 2;
    copyHeight[2] = cudaProducer->height / 2;
    uvOffset[1] = cudaProducer->width * cudaProducer->height;
    uvOffset[2] =
        uvOffset[1] + cudaProducer->width / 2 * cudaProducer->height / 2;
  }
  if (cudaProducer->pitchLinearOutput) {
    for (i = 0; i < surfNum; i++) {
      cuStatus =
          cuMemcpy(cudaPtr[i], (CUdeviceptr)(cudaProducer->pBuff + uvOffset[i]),
                   copyWidthInBytes[i] * copyHeight[i]);

      if (cuStatus != CUDA_SUCCESS) {
        printf("Cuda producer: cuMemCpy pitchlinear failed, cuStatus =%d\n",
               cuStatus);
        goto done;
      }
    }
  } else {
    // copy cudaProducer->pBuff to cudaArray
    CUDA_MEMCPY3D cpdesc;
    for (i = 0; i < surfNum; i++) {
      memset(&cpdesc, 0, sizeof(cpdesc));
      cpdesc.srcXInBytes = cpdesc.srcY = cpdesc.srcZ = cpdesc.srcLOD = 0;
      cpdesc.srcMemoryType = CU_MEMORYTYPE_HOST;
      cpdesc.srcHost = (void *)(cudaProducer->pBuff + uvOffset[i]);
      cpdesc.dstXInBytes = cpdesc.dstY = cpdesc.dstZ = cpdesc.dstLOD = 0;
      cpdesc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
      cpdesc.dstArray = cudaArr[i];
      cpdesc.WidthInBytes = copyWidthInBytes[i];
      cpdesc.Height = copyHeight[i];
      cpdesc.Depth = 1;
      cuStatus = cuMemcpy3D(&cpdesc);
      if (cuStatus != CUDA_SUCCESS) {
        printf("Cuda producer: cuMemCpy failed, cuStatus =%d\n", cuStatus);
        goto done;
      }
    }
  }
  for (i = 0; i < surfNum; i++) {
    if (cudaProducer->pitchLinearOutput)
      cudaEgl.frame.pPitch[i] = (void *)cudaPtr[i];
    else
      cudaEgl.frame.pArray[i] = cudaArr[i];
  }
  cudaEgl.width = copyWidthInBytes[0];
  cudaEgl.depth = 1;
  cudaEgl.height = copyHeight[0];
  cudaEgl.pitch = cudaProducer->pitchLinearOutput ? cudaEgl.width : 0;
  cudaEgl.frameType = cudaProducer->pitchLinearOutput ? CU_EGL_FRAME_TYPE_PITCH
                                                      : CU_EGL_FRAME_TYPE_ARRAY;
  cudaEgl.planeCount = surfNum;
  cudaEgl.numChannels = (eglColorFormat == CU_EGL_COLOR_FORMAT_ARGB) ? 4 : 1;
  cudaEgl.eglColorFormat = eglColorFormat;
  cudaEgl.cuFormat = CU_AD_FORMAT_UNSIGNED_INT8;

  static int numFramesPresented = 0;
  // If there is a frame presented before we check if consumer
  // is done with it using cuEGLStreamProducerReturnFrame.
  while (numFramesPresented) {
    CUeglFrame returnedCudaEgl;
    cuStatus = cuEGLStreamProducerReturnFrame(&cudaProducer->cudaConn,
                                              &returnedCudaEgl, NULL);
    if (cuStatus == CUDA_ERROR_LAUNCH_TIMEOUT) {
      continue;
    } else if (cuStatus != CUDA_SUCCESS) {
      printf("cuda Producer return frame FAILED with custatus= %d\n", cuStatus);
      return cuStatus;
    } else {
      numFramesPresented--;
    }
  }

  cuStatus =
      cuEGLStreamProducerPresentFrame(&cudaProducer->cudaConn, cudaEgl, NULL);
  if (cuStatus != CUDA_SUCCESS) {
    printf("cuda Producer present frame FAILED with custatus= %d\n", cuStatus);
    goto done;
  }
  numFramesPresented++;

done:
  if (file_p) {
    fclose(file_p);
    file_p = NULL;
  }

  return cuStatus;
}

CUresult cudaDeviceCreateProducer(test_cuda_producer_s *cudaProducer,
                                  CUdevice device) {
  CUresult status = CUDA_SUCCESS;
  if (CUDA_SUCCESS != (status = cuInit(0))) {
    printf("Failed to initialize CUDA\n");
    return status;
  }

  int major = 0, minor = 0;
  char deviceName[256];
  checkCudaErrors(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  checkCudaErrors(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  checkCudaErrors(cuDeviceGetName(deviceName, 256, device));
  printf(
      "CUDA Producer on GPU Device %d: \"%s\" with compute capability "
      "%d.%d\n\n",
      device, deviceName, major, minor);

  if (major < 6) {
    printf(
        "EGLStreams_CUDA_Interop requires SM 6.0 or higher arch GPU.  "
        "Exiting...\n");
    exit(2);  // EXIT_WAIVED
  }

  if (CUDA_SUCCESS !=
      (status = cuCtxCreate(&cudaProducer->context, 0, device))) {
    printf("failed to create CUDA context\n");
    return status;
  }

  status = cuMemAlloc(&cudaProducer->cudaPtrARGB[0], (WIDTH * HEIGHT * 4));
  if (status != CUDA_SUCCESS) {
    printf("Create CUDA pointer failed, cuStatus=%d\n", status);
    return status;
  }

  status = cuMemAlloc(&cudaProducer->cudaPtrYUV[0], (WIDTH * HEIGHT));
  if (status != CUDA_SUCCESS) {
    printf("Create CUDA pointer failed, cuStatus=%d\n", status);
    return status;
  }
  status = cuMemAlloc(&cudaProducer->cudaPtrYUV[1], (WIDTH * HEIGHT) / 4);
  if (status != CUDA_SUCCESS) {
    printf("Create CUDA pointer failed, cuStatus=%d\n", status);
    return status;
  }
  status = cuMemAlloc(&cudaProducer->cudaPtrYUV[2], (WIDTH * HEIGHT) / 4);
  if (status != CUDA_SUCCESS) {
    printf("Create CUDA pointer failed, cuStatus=%d\n", status);
    return status;
  }

  CUDA_ARRAY3D_DESCRIPTOR desc = {0};

  desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
  desc.Depth = 1;
  desc.Flags = CUDA_ARRAY3D_SURFACE_LDST;
  desc.NumChannels = 4;
  desc.Width = WIDTH * 4;
  desc.Height = HEIGHT;
  status = cuArray3DCreate(&cudaProducer->cudaArrARGB[0], &desc);
  if (status != CUDA_SUCCESS) {
    printf("Create CUDA array failed, cuStatus=%d\n", status);
    return status;
  }

  for (int i = 0; i < 3; i++) {
    if (i == 0) {
      desc.NumChannels = 1;
      desc.Width = WIDTH;
      desc.Height = HEIGHT;
    } else {  // U/V surface as planar
      desc.NumChannels = 1;
      desc.Width = WIDTH / 2;
      desc.Height = HEIGHT / 2;
    }
    status = cuArray3DCreate(&cudaProducer->cudaArrYUV[i], &desc);
    if (status != CUDA_SUCCESS) {
      printf("Create CUDA array failed, cuStatus=%d\n", status);
      return status;
    }
  }

  cudaProducer->pBuff = (unsigned char *)malloc((WIDTH * HEIGHT * 4));
  if (!cudaProducer->pBuff) {
    printf("CudaProducer: Failed to allocate image buffer\n");
  }

  checkCudaErrors(cuCtxPopCurrent(&cudaProducer->context));
  return status;
}

void cudaProducerInit(test_cuda_producer_s *cudaProducer, EGLDisplay eglDisplay,
                      EGLStreamKHR eglStream, TestArgs *args) {
  cudaProducer->fileName1 = args->infile1;
  cudaProducer->fileName2 = args->infile2;

  cudaProducer->frameCount = 2;
  cudaProducer->width = args->inputWidth;
  cudaProducer->height = args->inputHeight;
  cudaProducer->isARGB = args->isARGB;
  cudaProducer->pitchLinearOutput = args->pitchLinearOutput;

  // Set cudaProducer default parameters
  cudaProducer->eglDisplay = eglDisplay;
  cudaProducer->eglStream = eglStream;
}

CUresult cudaProducerDeinit(test_cuda_producer_s *cudaProducer) {
  if (cudaProducer->pBuff) free(cudaProducer->pBuff);

  checkCudaErrors(cuMemFree(cudaProducer->cudaPtrARGB[0]));
  checkCudaErrors(cuMemFree(cudaProducer->cudaPtrYUV[0]));
  checkCudaErrors(cuMemFree(cudaProducer->cudaPtrYUV[1]));
  checkCudaErrors(cuMemFree(cudaProducer->cudaPtrYUV[2]));
  checkCudaErrors(cuArrayDestroy(cudaProducer->cudaArrARGB[0]));
  checkCudaErrors(cuArrayDestroy(cudaProducer->cudaArrYUV[0]));
  checkCudaErrors(cuArrayDestroy(cudaProducer->cudaArrYUV[1]));
  checkCudaErrors(cuArrayDestroy(cudaProducer->cudaArrYUV[2]));

  return cuEGLStreamProducerDisconnect(&cudaProducer->cudaConn);
}
