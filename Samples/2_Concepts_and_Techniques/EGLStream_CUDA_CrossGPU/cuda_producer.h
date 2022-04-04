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
// DESCRIPTION:   Simple cuda producer header file
//

#ifndef _CUDA_PRODUCER_H_
#define _CUDA_PRODUCER_H_
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "cudaEGL.h"
#include "eglstrm_common.h"
#include <cuda_runtime.h>
#include <cuda.h>

typedef struct _test_cuda_producer_s {
  //  Stream params
  CUcontext context;
  CUeglStreamConnection cudaConn;
  int cudaDevId;
  EGLStreamKHR eglStream;
  EGLDisplay eglDisplay;
  unsigned int charCnt;
  bool profileAPI;
  char *tempBuff;
  CUdeviceptr cudaPtr;
  CUdeviceptr cudaPtr1;
  CUstream prodCudaStream;
} test_cuda_producer_s;

CUresult cudaProducerInit(test_cuda_producer_s *cudaProducer, TestArgs *args);
CUresult cudaProducerPresentFrame(test_cuda_producer_s *parserArg,
                                  CUeglFrame cudaEgl, int t);
CUresult cudaProducerReturnFrame(test_cuda_producer_s *parserArg,
                                 CUeglFrame cudaEgl, int t);
CUresult cudaProducerDeinit(test_cuda_producer_s *cudaProducer);
CUresult cudaDeviceCreateProducer(test_cuda_producer_s *cudaProducer);
cudaError_t cudaProducer_filter(CUstream cStream, char *pSrc, int width,
                                int height, char expectedVal, char newVal,
                                int frameNumber);
void cudaProducerPrepareFrame(CUeglFrame *cudaEgl, CUdeviceptr cudaPtr,
                              int bufferSize);
#endif
