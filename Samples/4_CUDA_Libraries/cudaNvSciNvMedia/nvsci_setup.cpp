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

#include "nvmedia_utils/cmdline.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "nvsci_setup.h"
#include "nvmedia_2d_nvscisync.h"

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

void setupNvMediaSignalerNvSciSync(Blit2DTest *ctx, NvSciSyncObj &syncObj,
                                   int cudaDeviceId) {
  NvSciSyncModule sciSyncModule;
  checkNvSciErrors(NvSciSyncModuleOpen(&sciSyncModule));
  NvSciSyncAttrList signalerAttrList, waiterAttrList;
  NvSciSyncAttrList syncUnreconciledList[2];
  NvSciSyncAttrList syncReconciledList, syncConflictList;

  checkNvSciErrors(NvSciSyncAttrListCreate(sciSyncModule, &signalerAttrList));
  checkNvSciErrors(NvSciSyncAttrListCreate(sciSyncModule, &waiterAttrList));

  NvMediaStatus status = NvMedia2DFillNvSciSyncAttrList(
      ctx->i2d, signalerAttrList, NVMEDIA_SIGNALER);
  if (status != NVMEDIA_STATUS_OK) {
    printf("%s: NvMedia2DFillNvSciSyncAttrList failed\n", __func__);
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaSetDevice(cudaDeviceId));
  checkCudaErrors(cudaDeviceGetNvSciSyncAttributes(waiterAttrList, cudaDeviceId,
                                                   cudaNvSciSyncAttrWait));

  syncUnreconciledList[0] = signalerAttrList;
  syncUnreconciledList[1] = waiterAttrList;
  checkNvSciErrors(NvSciSyncAttrListReconcile(
      syncUnreconciledList, 2, &syncReconciledList, &syncConflictList));
  checkNvSciErrors(NvSciSyncObjAlloc(syncReconciledList, &syncObj));

  NvSciSyncAttrListFree(signalerAttrList);
  NvSciSyncAttrListFree(waiterAttrList);
  if (syncConflictList != nullptr) {
    NvSciSyncAttrListFree(syncConflictList);
  }
}

void setupCudaSignalerNvSciSync(Blit2DTest *ctx, NvSciSyncObj &syncObj,
                                int cudaDeviceId) {
  NvSciSyncModule sciSyncModule;
  checkNvSciErrors(NvSciSyncModuleOpen(&sciSyncModule));
  NvSciSyncAttrList signalerAttrList, waiterAttrList;
  NvSciSyncAttrList syncUnreconciledList[2];
  NvSciSyncAttrList syncReconciledList, syncConflictList;

  checkNvSciErrors(NvSciSyncAttrListCreate(sciSyncModule, &signalerAttrList));
  checkNvSciErrors(NvSciSyncAttrListCreate(sciSyncModule, &waiterAttrList));

  NvMediaStatus status =
      NvMedia2DFillNvSciSyncAttrList(ctx->i2d, waiterAttrList, NVMEDIA_WAITER);
  if (status != NVMEDIA_STATUS_OK) {
    printf("%s: NvMedia2DFillNvSciSyncAttrList failed\n", __func__);
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaSetDevice(cudaDeviceId));
  checkCudaErrors(cudaDeviceGetNvSciSyncAttributes(
      signalerAttrList, cudaDeviceId, cudaNvSciSyncAttrSignal));

  syncUnreconciledList[0] = signalerAttrList;
  syncUnreconciledList[1] = waiterAttrList;
  checkNvSciErrors(NvSciSyncAttrListReconcile(
      syncUnreconciledList, 2, &syncReconciledList, &syncConflictList));
  checkNvSciErrors(NvSciSyncObjAlloc(syncReconciledList, &syncObj));

  NvSciSyncAttrListFree(signalerAttrList);
  NvSciSyncAttrListFree(waiterAttrList);
  if (syncConflictList != nullptr) {
    NvSciSyncAttrListFree(syncConflictList);
  }
}

void setupNvSciBuf(NvSciBufObj &bufobj, NvSciBufAttrList &nvmediaAttrlist,
                   int cudaDeviceId) {
  CUuuid devUUID;
  NvSciBufAttrList conflictlist;
  NvSciBufAttrList bufUnreconciledAttrlist[1];

  CUresult res = cuDeviceGetUuid(&devUUID, cudaDeviceId);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "Driver API error = %04d \n", res);
    exit(EXIT_FAILURE);
  }

  NvSciBufAttrKeyValuePair attr_gpuid[] = {NvSciBufGeneralAttrKey_GpuId,
                                           &devUUID, sizeof(devUUID)};

  // set CUDA GPU ID to attribute list
  checkNvSciErrors(NvSciBufAttrListSetAttrs(
      nvmediaAttrlist, attr_gpuid,
      sizeof(attr_gpuid) / sizeof(NvSciBufAttrKeyValuePair)));

  bufUnreconciledAttrlist[0] = nvmediaAttrlist;

  checkNvSciErrors(NvSciBufAttrListReconcileAndObjAlloc(
      bufUnreconciledAttrlist, 1, &bufobj, &conflictlist));
  if (conflictlist != NULL) {
    NvSciBufAttrListFree(conflictlist);
  }
}

void cleanupNvSciBuf(NvSciBufObj &Bufobj) {
  if (Bufobj != NULL) {
    NvSciBufObjFree(Bufobj);
  }
}

void cleanupNvSciSync(NvSciSyncObj &syncObj) {
  if (NvSciSyncObjFree != NULL) {
    NvSciSyncObjFree(syncObj);
  }
}
