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

#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "resize_convert.h"
#include "utils.h"

__global__ void floatToChar(float *src, unsigned char *dst, int height,
                            int width, int batchSize) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if (x >= height * width) return;

  int offset = height * width * 3;

  for (int j = 0; j < batchSize; j++) {
    // b
    *(dst + j * offset + x * 3 + 0) =
        (unsigned char)*(src + j * offset + height * width * 0 + x);
    // g
    *(dst + j * offset + x * 3 + 1) =
        (unsigned char)*(src + j * offset + height * width * 1 + x);
    // r
    *(dst + j * offset + x * 3 + 2) =
        (unsigned char)*(src + j * offset + height * width * 2 + x);
  }
}

void floatPlanarToChar(float *src, unsigned char *dst, int height, int width,
                       int batchSize) {
  floatToChar<<<(height * width - 1) / 1024 + 1, 1024, 0, NULL>>>(
      src, dst, height, width, batchSize);
}

void dumpRawBGR(float *d_srcBGR, int pitch, int width, int height,
                int batchSize, char *folder, char *tag) {
  float *bgr, *d_bgr;
  int frameSize;
  char directory[120];
  char mkdir_cmd[256];
#if !defined(_WIN32)
  sprintf(directory, "output/%s", folder);
  sprintf(mkdir_cmd, "mkdir -p %s 2> /dev/null", directory);
#else
  sprintf(directory, "output\\%s", folder);
  sprintf(mkdir_cmd, "mkdir %s 2> nul", directory);
#endif

  int ret = system(mkdir_cmd);

  frameSize = width * height * 3 * sizeof(float);
  bgr = (float *)malloc(frameSize);
  if (bgr == NULL) {
    std::cerr << "Failed malloc for bgr\n";
    return;
  }

  d_bgr = d_srcBGR;
  for (int i = 0; i < batchSize; i++) {
    char filename[120];
    std::ofstream *outputFile;

    checkCudaErrors(cudaMemcpy((void *)bgr, (void *)d_bgr, frameSize,
                               cudaMemcpyDeviceToHost));
    sprintf(filename, "%s/%s_%d.raw", directory, tag, (i + 1));

    outputFile = new std::ofstream(filename);
    if (outputFile) {
      outputFile->write((char *)bgr, frameSize);
      delete outputFile;
    }

    d_bgr += pitch * height * 3;
  }

  free(bgr);
}

void dumpBGR(float *d_srcBGR, int pitch, int width, int height, int batchSize,
             char *folder, char *tag) {
  dumpRawBGR(d_srcBGR, pitch, width, height, batchSize, folder, tag);
}

void dumpYUV(unsigned char *d_nv12, int size, char *folder, char *tag) {
  unsigned char *nv12Data;
  std::ofstream *nv12File;
  char filename[120];
  char directory[60];
  char mkdir_cmd[256];
#if !defined(_WIN32)
  sprintf(directory, "output/%s", folder);
  sprintf(mkdir_cmd, "mkdir -p %s 2> /dev/null", directory);
#else
  sprintf(directory, "output\\%s", folder);
  sprintf(mkdir_cmd, "mkdir %s 2> nul", directory);
#endif

  int ret = system(mkdir_cmd);

  sprintf(filename, "%s/%s.nv12", directory, tag);

  nv12File = new std::ofstream(filename);
  if (nv12File == NULL) {
    std::cerr << "Failed to new " << filename;
    return;
  }

  nv12Data = (unsigned char *)malloc(size * (sizeof(char)));
  if (nv12Data == NULL) {
    std::cerr << "Failed to allcoate memory\n";
    return;
  }

  cudaMemcpy((void *)nv12Data, (void *)d_nv12, size, cudaMemcpyDeviceToHost);

  nv12File->write((const char *)nv12Data, size);

  free(nv12Data);
  delete nv12File;
}
