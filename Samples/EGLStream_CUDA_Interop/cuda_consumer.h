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
// DESCRIPTION:   CUDA consumer header file
//

#ifndef _CUDA_CONSUMER_H_
#define _CUDA_CONSUMER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cudaEGL.h"
#include "eglstrm_common.h"

extern EGLStreamKHR eglStream;
extern EGLDisplay g_display;

typedef struct _test_cuda_consumer_s {
  CUcontext context;
  CUeglStreamConnection cudaConn;
  bool pitchLinearOutput;
  unsigned int width;
  unsigned int height;
  char *fileName1;
  char *fileName2;
  char *outFile1;
  char *outFile2;
  unsigned int frameCount;
} test_cuda_consumer_s;

void cuda_consumer_init(test_cuda_consumer_s *cudaConsumer, TestArgs *args);
CUresult cuda_consumer_deinit(test_cuda_consumer_s *cudaConsumer);
CUresult cudaConsumerTest(test_cuda_consumer_s *data, char *outFile);
CUresult cudaDeviceCreateConsumer(test_cuda_consumer_s *cudaConsumer,
                                  CUdevice device);
#endif
