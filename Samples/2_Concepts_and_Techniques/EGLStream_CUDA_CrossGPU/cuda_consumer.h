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
// DESCRIPTION:   CUDA consumer header file
//

#ifndef _CUDA_CONSUMER_H_
#define _CUDA_CONSUMER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cudaEGL.h"
#include "eglstrm_common.h"
#include <cuda_runtime.h>
#include <cuda.h>

typedef struct _test_cuda_consumer_s {
  CUcontext context;
  CUeglStreamConnection cudaConn;
  int cudaDevId;
  EGLDisplay eglDisplay;
  EGLStreamKHR eglStream;
  unsigned int charCnt;
  char *cudaBuf;
  bool profileAPI;
  unsigned char *pCudaCopyMem;
  CUstream consCudaStream;
} test_cuda_consumer_s;

CUresult cuda_consumer_init(test_cuda_consumer_s *cudaConsumer, TestArgs *args);
CUresult cuda_consumer_Deinit(test_cuda_consumer_s *cudaConsumer);
CUresult cudaConsumerAcquireFrame(test_cuda_consumer_s *data, int frameNumber);
CUresult cudaConsumerReleaseFrame(test_cuda_consumer_s *data, int frameNumber);
CUresult cudaDeviceCreateConsumer(test_cuda_consumer_s *cudaConsumer);
cudaError_t cudaConsumer_filter(CUstream cStream, char *pSrc, int width,
                                int height, char expectedVal, char newVal,
                                int frameNumber);
cudaError_t cudaGetValueMismatch(void);

#endif
