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

#ifndef CUDANVSCI_H
#define CUDANVSCI_H

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <nvscibuf.h>
#include <nvscisync.h>
#include <vector>

#define checkNvSciErrors(call)                              \
  do {                                                      \
    NvSciError _status = call;                              \
    if (NvSciError_Success != _status) {                    \
      printf(                                               \
          "NVSCI call in file '%s' in line %i returned"     \
          " %d, expected %d\n",                             \
          __FILE__, __LINE__, _status, NvSciError_Success); \
      fflush(stdout);                                       \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  } while (0)

extern void rotateKernel(cudaTextureObject_t &texObj, const float angle,
                         unsigned int *d_outputData, const int imageWidth,
                         const int imageHeight, cudaStream_t stream);
extern void launchGrayScaleKernel(unsigned int *d_rgbaImage,
                                  std::string image_filename, size_t imageWidth,
                                  size_t imageHeight, cudaStream_t stream);

class cudaNvSci {
 private:
  int m_isMultiGPU;
  int m_cudaNvSciSignalDeviceId;
  int m_cudaNvSciWaitDeviceId;
  unsigned char *image_data;
  size_t m_bufSize;
  size_t imageWidth;
  size_t imageHeight;

 public:
  NvSciSyncModule syncModule;
  NvSciBufModule buffModule;
  NvSciSyncAttrList syncUnreconciledList[2];
  NvSciSyncAttrList syncReconciledList;
  NvSciSyncAttrList syncConflictList;

  NvSciBufAttrList rawBufUnreconciledList[2];
  NvSciBufAttrList imageBufUnreconciledList[2];
  NvSciBufAttrList rawBufReconciledList;
  NvSciBufAttrList buffConflictList;
  NvSciBufAttrList imageBufReconciledList;
  NvSciBufAttrList imageBufConflictList;
  NvSciBufAttrList buffAttrListOut;

  NvSciSyncObj syncObj;
  NvSciBufObj rawBufObj;
  NvSciBufObj imageBufObj;
  NvSciSyncFence *fence;

  cudaNvSci(int isMultiGPU, std::vector<int> &deviceIds,
            unsigned char *image_data, size_t imageWidth, size_t imageHeight);

  void initNvSci();

  void runCudaNvSci(std::string &image_filename);

  void createNvSciRawBufObj();

  void createNvSciSyncObj();

  void createNvSciBufImageObj();
};

#endif  // CUDANVSCI_H
