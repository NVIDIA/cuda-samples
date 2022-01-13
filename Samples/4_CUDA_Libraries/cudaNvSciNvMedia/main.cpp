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

/* standard headers */
#include <string.h>
#include <iostream>
#include <signal.h>
#include <thread>

/* Nvidia headers */
#include <nvscisync.h>
#include "nvmedia_utils/cmdline.h"
#include "nvmedia_image.h"
#include "nvmedia_2d.h"
#include "nvmedia_2d_nvscisync.h"
#include "nvmedia_surface.h"
#include "nvmedia_utils/image_utils.h"
#include "nvmedia_image_nvscibuf.h"
#include "cuda_consumer.h"
#include "nvmedia_producer.h"
#include "nvsci_setup.h"

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

static void cleanup(Blit2DTest* ctx, NvMediaStatus status) {
  if (ctx->i2d != NULL) {
    NvMedia2DDestroy(ctx->i2d);
  }

  if (ctx->device != NULL) {
    NvMediaDeviceDestroy(ctx->device);
  }
  if (status != NVMEDIA_STATUS_OK) {
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[]) {
  TestArgs args;
  Blit2DTest ctx;
  NvMediaStatus status = NVMEDIA_STATUS_ERROR;
  NvSciSyncFence nvMediaSignalerFence = NvSciSyncFenceInitializer;
  NvSciSyncFence cudaSignalerFence = NvSciSyncFenceInitializer;

  int cudaDeviceId;
  uint64_t startTime, endTime;
  uint64_t operationStartTime, operationEndTime;
  double processingTime;

  /* Read configuration from command line and config file */
  memset(&args, 0, sizeof(TestArgs));
  memset(&ctx, 0, sizeof(Blit2DTest));

  /* ParseArgs parses the command line and the 2D configuration file and
   * populates all initParams and run time configuration in to appropriate
   * structures within args
   */
  if (ParseArgs(argc, argv, &args)) {
    PrintUsage();
    return -1;
  }
  /* Check version */
  NvMediaVersion version;
  status = NvMedia2DGetVersion(&version);
  if (status == NVMEDIA_STATUS_OK) {
    printf("Library version: %u.%u\n", version.major, version.minor);
    printf("Header version:  %u.%u\n", NVMEDIA_2D_VERSION_MAJOR,
           NVMEDIA_2D_VERSION_MINOR);
    if ((version.major != NVMEDIA_2D_VERSION_MAJOR) ||
        (version.minor != NVMEDIA_2D_VERSION_MINOR)) {
      printf("Library and Header mismatch!\n");
      cleanup(&ctx, status);
    }
  }

  // Create NvMedia device
  ctx.device = NvMediaDeviceCreate();
  if (!ctx.device) {
    printf("%s: Failed to create NvMedia device\n", __func__);
    cleanup(&ctx, status);
  }

  // Create 2D blitter
  ctx.i2d = NvMedia2DCreate(ctx.device);
  if (!ctx.i2d) {
    printf("%s: Failed to create NvMedia 2D i2d\n", __func__);
    cleanup(&ctx, status);
  }

  cudaDeviceId = findCudaDevice(argc, (const char**)argv);

  // NvMedia-CUDA operations without NvSCI APIs starts
  cudaResources cudaResObj;
  GetTimeMicroSec(&startTime);
  setupNvMedia(&args, &ctx);
  setupCuda(&ctx, cudaResObj, cudaDeviceId);

  GetTimeMicroSec(&operationStartTime);
  for (int i = 0; i < args.iterations; i++) {
    runNvMediaBlit2D(&args, &ctx);
    runCudaOperation(&ctx, cudaResObj, cudaDeviceId);
  }
  GetTimeMicroSec(&operationEndTime);

  cleanupNvMedia(&ctx);
  cleanupCuda(&ctx, cudaResObj);
  GetTimeMicroSec(&endTime);
  // NvMedia-CUDA operations without NvSCI APIs ends

  processingTime = (double)(operationEndTime - operationStartTime) / 1000.0;
  printf(
      "Overall Processing time of NvMedia-CUDA Operations without NvSCI APIs "
      "%.4f ms  with %zu iterations\n",
      processingTime, args.iterations);
  processingTime = (double)(endTime - startTime) / 1000.0;
  printf(
      "Overall Processing time of NvMedia-CUDA Operations + allocation/cleanup "
      "without NvSCI APIs %.4f ms  with %zu iterations\n",
      processingTime, args.iterations);

  NvSciBufObj dstNvSciBufobj, srcNvSciBufobj;
  NvSciSyncObj nvMediaSignalerSyncObj, cudaSignalerSyncObj;
  cudaExternalResInterop cudaExtResObj;
  // NvMedia-CUDA operations via interop with NvSCI APIs starts
  GetTimeMicroSec(&startTime);
  setupNvMediaSignalerNvSciSync(&ctx, nvMediaSignalerSyncObj, cudaDeviceId);
  setupCudaSignalerNvSciSync(&ctx, cudaSignalerSyncObj, cudaDeviceId);
  setupNvMedia(&args, &ctx, srcNvSciBufobj, dstNvSciBufobj,
               nvMediaSignalerSyncObj, cudaSignalerSyncObj, cudaDeviceId);
  setupCuda(cudaExtResObj, dstNvSciBufobj, nvMediaSignalerSyncObj,
            cudaSignalerSyncObj, cudaDeviceId);

  GetTimeMicroSec(&operationStartTime);
  for (int i = 0; i < args.iterations; i++) {
    runNvMediaBlit2D(&args, &ctx, nvMediaSignalerSyncObj, &cudaSignalerFence,
                     &nvMediaSignalerFence);
    runCudaOperation(cudaExtResObj, &nvMediaSignalerFence, &cudaSignalerFence,
                     cudaDeviceId, args.iterations);
  }
  GetTimeMicroSec(&operationEndTime);

  cleanupNvMedia(&ctx, nvMediaSignalerSyncObj, cudaSignalerSyncObj);
  cleanupCuda(cudaExtResObj);
  cleanupNvSciSync(nvMediaSignalerSyncObj);
  cleanupNvSciSync(cudaSignalerSyncObj);
  cleanupNvSciBuf(srcNvSciBufobj);
  cleanupNvSciBuf(dstNvSciBufobj);
  GetTimeMicroSec(&endTime);
  // NvMedia-CUDA operations via interop with NvSCI APIs ends

  processingTime = (double)(operationEndTime - operationStartTime) / 1000.0;
  printf(
      "Overall Processing time of NvMedia-CUDA Operations with NvSCI APIs %.4f "
      "ms  with %zu iterations\n",
      processingTime, args.iterations);
  processingTime = (double)(endTime - startTime) / 1000.0;
  printf(
      "Overall Processing time of NvMedia-CUDA Operations + allocation/cleanup "
      "with NvSCI APIs %.4f ms  with %zu iterations\n",
      processingTime, args.iterations);

  if (ctx.i2d != NULL) {
    NvMedia2DDestroy(ctx.i2d);
  }

  if (ctx.device != NULL) {
    NvMediaDeviceDestroy(ctx.device);
  }

  if (status == NVMEDIA_STATUS_OK) {
    return 0;
  } else {
    return 1;
  }
}
