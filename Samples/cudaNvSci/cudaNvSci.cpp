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

#include "cudaNvSci.h"
#include <cuda.h>
#include <condition_variable>
#include <iostream>
#include <thread>

std::mutex m_mutex;
std::condition_variable m_condVar;
bool workSubmitted = false;

class cudaNvSciSignal {
 private:
  NvSciSyncModule m_syncModule;
  NvSciBufModule m_bufModule;

  NvSciSyncAttrList m_syncAttrList;
  NvSciSyncFence *m_fence;

  NvSciBufAttrList m_rawBufAttrList;
  NvSciBufAttrList m_imageBufAttrList;
  NvSciBufAttrList m_buffAttrListOut[2];
  NvSciBufAttrKeyValuePair pairArrayOut[10];

  cudaExternalMemory_t extMemRawBuf, extMemImageBuf;
  cudaMipmappedArray_t d_mipmapArray;
  cudaArray_t d_mipLevelArray;
  cudaTextureObject_t texObject;
  cudaExternalSemaphore_t signalSem;

  cudaStream_t streamToRun;
  int m_cudaDeviceId;
  CUuuid m_devUUID;
  uint64_t m_imageWidth;
  uint64_t m_imageHeight;
  void *d_outputBuf;
  size_t m_bufSize;

 public:
  cudaNvSciSignal(NvSciBufModule bufModule, NvSciSyncModule syncModule,
                  int cudaDeviceId, int bufSize, uint64_t imageWidth,
                  uint64_t imageHeight, NvSciSyncFence *fence)
      : m_syncModule(syncModule),
        m_bufModule(bufModule),
        m_cudaDeviceId(cudaDeviceId),
        d_outputBuf(NULL),
        m_bufSize(bufSize),
        m_imageWidth(imageWidth),
        m_imageHeight(imageHeight),
        m_fence(fence) {
    initCuda();

    checkNvSciErrors(NvSciSyncAttrListCreate(m_syncModule, &m_syncAttrList));
    checkNvSciErrors(NvSciBufAttrListCreate(m_bufModule, &m_rawBufAttrList));
    checkNvSciErrors(NvSciBufAttrListCreate(m_bufModule, &m_imageBufAttrList));

    setRawBufAttrList(m_bufSize);
    setImageBufAttrList(m_imageWidth, m_imageHeight);

    checkCudaErrors(cudaDeviceGetNvSciSyncAttributes(
        m_syncAttrList, m_cudaDeviceId, cudaNvSciSyncAttrSignal));
  }

  ~cudaNvSciSignal() {
    checkCudaErrors(cudaSetDevice(m_cudaDeviceId));
    checkCudaErrors(cudaFreeMipmappedArray(d_mipmapArray));
    checkCudaErrors(cudaFree(d_outputBuf));
    checkCudaErrors(cudaDestroyExternalSemaphore(signalSem));
    checkCudaErrors(cudaDestroyExternalMemory(extMemRawBuf));
    checkCudaErrors(cudaDestroyExternalMemory(extMemImageBuf));
    checkCudaErrors(cudaDestroyTextureObject(texObject));
    checkCudaErrors(cudaStreamDestroy(streamToRun));
  }

  void initCuda() {
    checkCudaErrors(cudaSetDevice(m_cudaDeviceId));
    checkCudaErrors(
        cudaStreamCreateWithFlags(&streamToRun, cudaStreamNonBlocking));

    int major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, m_cudaDeviceId));
    checkCudaErrors(cudaDeviceGetAttribute(
        &minor, cudaDevAttrComputeCapabilityMinor, m_cudaDeviceId));
    printf(
        "[cudaNvSciSignal] GPU Device %d: \"%s\" with compute capability "
        "%d.%d\n\n",
        m_cudaDeviceId, _ConvertSMVer2ArchName(major, minor), major, minor);

    CUresult res = cuDeviceGetUuid(&m_devUUID, m_cudaDeviceId);
    if (res != CUDA_SUCCESS) {
      fprintf(stderr, "Driver API error = %04d \n", res);
      exit(EXIT_FAILURE);
    }
  }

  void setRawBufAttrList(uint64_t size) {
    NvSciBufType bufType = NvSciBufType_RawBuffer;
    uint64_t tempAlignment = 0;
    bool cpuAccess = false;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrKeyValuePair rawBufAttrs[] = {
        {NvSciBufRawBufferAttrKey_Size, &size, sizeof(size)},
        {NvSciBufRawBufferAttrKey_Align, &tempAlignment, sizeof(tempAlignment)},
        {NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess)},
        {NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
        {NvSciBufGeneralAttrKey_GpuId, &m_devUUID, sizeof(m_devUUID)},
    };

    checkNvSciErrors(NvSciBufAttrListSetAttrs(
        m_rawBufAttrList, rawBufAttrs,
        sizeof(rawBufAttrs) / sizeof(NvSciBufAttrKeyValuePair)));
  }

  void setImageBufAttrList(uint32_t width, uint32_t height) {
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_BlockLinearType;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;

    uint32_t planeCount = 1;
    uint32_t planeWidths[] = {width};
    uint32_t planeHeights[] = {height};
    uint64_t lrpad = 0, tbpad = 100;

    bool cpuAccessFlag = false;

    NvSciBufAttrValColorFmt planecolorfmts[] = {NvSciColor_B8G8R8A8};
    NvSciBufAttrValColorStd planecolorstds[] = {NvSciColorStd_SRGB};
    NvSciBufAttrValImageScanType planescantype[] = {NvSciBufScan_InterlaceType};

    NvSciBufAttrKeyValuePair imgBufAttrs[] = {
        {NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
        {NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount)},
        {NvSciBufImageAttrKey_Layout, &layout, sizeof(layout)},
        {NvSciBufImageAttrKey_TopPadding, &tbpad, sizeof(tbpad)},
        {NvSciBufImageAttrKey_BottomPadding, &tbpad, sizeof(tbpad)},
        {NvSciBufImageAttrKey_LeftPadding, &lrpad, sizeof(lrpad)},
        {NvSciBufImageAttrKey_RightPadding, &lrpad, sizeof(lrpad)},
        {NvSciBufImageAttrKey_PlaneColorFormat, planecolorfmts,
         sizeof(planecolorfmts)},
        {NvSciBufImageAttrKey_PlaneColorStd, planecolorstds,
         sizeof(planecolorstds)},
        {NvSciBufImageAttrKey_PlaneWidth, planeWidths, sizeof(planeWidths)},
        {NvSciBufImageAttrKey_PlaneHeight, planeHeights, sizeof(planeHeights)},
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccessFlag,
         sizeof(cpuAccessFlag)},
        {NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
        {NvSciBufImageAttrKey_PlaneScanType, planescantype,
         sizeof(planescantype)},
        {NvSciBufGeneralAttrKey_GpuId, &m_devUUID, sizeof(m_devUUID)},
    };

    checkNvSciErrors(NvSciBufAttrListSetAttrs(
        m_imageBufAttrList, imgBufAttrs,
        sizeof(imgBufAttrs) / sizeof(NvSciBufAttrKeyValuePair)));
  }

  NvSciSyncAttrList getNvSciSyncAttrList() { return m_syncAttrList; }

  NvSciBufAttrList getNvSciRawBufAttrList() { return m_rawBufAttrList; }

  NvSciBufAttrList getNvSciImageBufAttrList() { return m_imageBufAttrList; }

  void runRotateImageAndSignal(unsigned char *imageData) {
    int numOfGPUs = 0;
    checkCudaErrors(cudaGetDeviceCount(&numOfGPUs));  // For cuda init purpose
    checkCudaErrors(cudaSetDevice(m_cudaDeviceId));

    copyDataToImageArray(imageData);
    createTexture();

    float angle = 0.5f;  // angle to rotate image by (in radians)
    rotateKernel(texObject, angle, (unsigned int *)d_outputBuf, m_imageWidth,
                 m_imageHeight, streamToRun);

    signalExternalSemaphore();
  }

  void cudaImportNvSciSemaphore(NvSciSyncObj syncObj) {
    checkCudaErrors(cudaSetDevice(m_cudaDeviceId));

    cudaExternalSemaphoreHandleDesc extSemDesc;
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDesc.handle.nvSciSyncObj = (void *)syncObj;

    checkCudaErrors(cudaImportExternalSemaphore(&signalSem, &extSemDesc));
  }

  void signalExternalSemaphore() {
    cudaExternalSemaphoreSignalParams signalParams;
    memset(&signalParams, 0, sizeof(signalParams));
    // For cross-process signaler-waiter applications need to use NvSciIpc
    // and NvSciSync[Export|Import] utilities to share the NvSciSyncFence
    // across process. This step is optional in single-process.
    signalParams.params.nvSciSync.fence = (void *)m_fence;
    signalParams.flags = 0;

    checkCudaErrors(cudaSignalExternalSemaphoresAsync(&signalSem, &signalParams,
                                                      1, streamToRun));
  }

  void cudaImportNvSciRawBuf(NvSciBufObj inputBufObj) {
    checkCudaErrors(cudaSetDevice(m_cudaDeviceId));
    checkNvSciErrors(
        NvSciBufObjGetAttrList(inputBufObj, &m_buffAttrListOut[0]));

    memset(pairArrayOut, 0, sizeof(NvSciBufAttrKeyValuePair) * 10);
    pairArrayOut[0].key = NvSciBufRawBufferAttrKey_Size;
    pairArrayOut[1].key = NvSciBufRawBufferAttrKey_Align;

    checkNvSciErrors(
        NvSciBufAttrListGetAttrs(m_buffAttrListOut[0], pairArrayOut, 2));

    uint64_t size = *(uint64_t *)pairArrayOut[0].value;
    uint64_t offset = *(uint64_t *)pairArrayOut[1].value;

    cudaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = inputBufObj;
    memHandleDesc.size = size;
    checkCudaErrors(cudaImportExternalMemory(&extMemRawBuf, &memHandleDesc));

    cudaExternalMemoryBufferDesc bufferDesc;
    memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.offset = offset;
    bufferDesc.size = size;
    m_bufSize = size;
    checkCudaErrors(cudaExternalMemoryGetMappedBuffer(
        &d_outputBuf, extMemRawBuf, &bufferDesc));
  }

  void cudaImportNvSciImage(NvSciBufObj inputBufObj) {
    checkCudaErrors(cudaSetDevice(m_cudaDeviceId));
    checkNvSciErrors(
        NvSciBufObjGetAttrList(inputBufObj, &m_buffAttrListOut[1]));

    memset(pairArrayOut, 0, sizeof(NvSciBufAttrKeyValuePair) * 10);
    pairArrayOut[0].key = NvSciBufImageAttrKey_Size;
    pairArrayOut[1].key = NvSciBufImageAttrKey_Alignment;
    pairArrayOut[2].key = NvSciBufImageAttrKey_PlaneCount;
    pairArrayOut[3].key = NvSciBufImageAttrKey_PlaneWidth;
    pairArrayOut[4].key = NvSciBufImageAttrKey_PlaneHeight;

    checkNvSciErrors(
        NvSciBufAttrListGetAttrs(m_buffAttrListOut[1], pairArrayOut, 5));

    uint64_t size = *(uint64_t *)pairArrayOut[0].value;
    uint64_t alignment = *(uint64_t *)pairArrayOut[1].value;
    uint64_t planeCount = *(uint64_t *)pairArrayOut[2].value;
    uint64_t imageWidth = *(uint64_t *)pairArrayOut[3].value;
    uint64_t imageHeight = *(uint64_t *)pairArrayOut[4].value;

    cudaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = inputBufObj;
    memHandleDesc.size = size;
    checkCudaErrors(cudaImportExternalMemory(&extMemImageBuf, &memHandleDesc));

    cudaExtent extent = {};
    memset(&extent, 0, sizeof(extent));
    extent.width = imageWidth;
    extent.height = imageHeight;
    extent.depth = 0;

    cudaChannelFormatDesc desc;
    desc.x = 8;
    desc.y = 8;
    desc.z = 8;
    desc.w = 8;
    desc.f = cudaChannelFormatKindUnsigned;

    cudaExternalMemoryMipmappedArrayDesc mipmapDesc = {0};
    mipmapDesc.offset = 0;
    mipmapDesc.formatDesc = desc;
    mipmapDesc.extent = extent;
    mipmapDesc.flags = 0;

    mipmapDesc.numLevels = 1;
    checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(
        &d_mipmapArray, extMemImageBuf, &mipmapDesc));
  }

  void copyDataToImageArray(unsigned char *imageData) {
    uint32_t mipLevelId = 0;
    checkCudaErrors(cudaGetMipmappedArrayLevel(&d_mipLevelArray, d_mipmapArray,
                                               mipLevelId));

    checkCudaErrors(cudaMemcpy2DToArrayAsync(
        d_mipLevelArray, 0, 0, imageData, m_imageWidth * sizeof(unsigned int),
        m_imageWidth * sizeof(unsigned int), m_imageHeight,
        cudaMemcpyHostToDevice, streamToRun));
  }

  void createTexture() {
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_mipLevelArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeNormalizedFloat;

    checkCudaErrors(
        cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));
  }
};

class cudaNvSciWait {
 private:
  NvSciSyncModule m_syncModule;
  NvSciBufModule m_bufModule;

  NvSciSyncAttrList m_syncAttrList;
  NvSciBufAttrList m_rawBufAttrList;
  NvSciBufAttrList m_buffAttrListOut;
  NvSciBufAttrKeyValuePair pairArrayOut[10];
  NvSciSyncFence *m_fence;

  cudaExternalMemory_t extMemRawBuf;
  cudaExternalSemaphore_t waitSem;
  cudaStream_t streamToRun;
  int m_cudaDeviceId;
  CUuuid m_devUUID;
  void *d_outputBuf;
  size_t m_bufSize;
  size_t imageWidth;
  size_t imageHeight;

 public:
  cudaNvSciWait(NvSciBufModule bufModule, NvSciSyncModule syncModule,
                int cudaDeviceId, int bufSize, NvSciSyncFence *fence)
      : m_bufModule(bufModule),
        m_syncModule(syncModule),
        m_cudaDeviceId(cudaDeviceId),
        m_bufSize(bufSize),
        m_fence(fence) {
    initCuda();
    checkNvSciErrors(NvSciSyncAttrListCreate(m_syncModule, &m_syncAttrList));
    checkNvSciErrors(NvSciBufAttrListCreate(m_bufModule, &m_rawBufAttrList));

    setRawBufAttrList(m_bufSize);
    checkCudaErrors(cudaDeviceGetNvSciSyncAttributes(
        m_syncAttrList, m_cudaDeviceId, cudaNvSciSyncAttrWait));
  }

  ~cudaNvSciWait() {
    checkCudaErrors(cudaStreamDestroy(streamToRun));
    checkCudaErrors(cudaDestroyExternalSemaphore(waitSem));
    checkCudaErrors(cudaDestroyExternalMemory(extMemRawBuf));
    checkCudaErrors(cudaFree(d_outputBuf));
  }

  void initCuda() {
    checkCudaErrors(cudaSetDevice(m_cudaDeviceId));
    checkCudaErrors(
        cudaStreamCreateWithFlags(&streamToRun, cudaStreamNonBlocking));
    CUresult res = cuDeviceGetUuid(&m_devUUID, m_cudaDeviceId);
    if (res != CUDA_SUCCESS) {
      fprintf(stderr, "Driver API error = %04d \n", res);
      exit(EXIT_FAILURE);
    }

    int major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, m_cudaDeviceId));
    checkCudaErrors(cudaDeviceGetAttribute(
        &minor, cudaDevAttrComputeCapabilityMinor, m_cudaDeviceId));
    printf(
        "[cudaNvSciWait] GPU Device %d: \"%s\" with compute capability "
        "%d.%d\n\n",
        m_cudaDeviceId, _ConvertSMVer2ArchName(major, minor), major, minor);
  }

  void setRawBufAttrList(uint64_t size) {
    NvSciBufType bufType = NvSciBufType_RawBuffer;
    uint64_t tempAlignment = 0;
    bool cpuAccess = false;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrKeyValuePair rawBufAttrs[] = {
        {NvSciBufRawBufferAttrKey_Size, &size, sizeof(size)},
        {NvSciBufRawBufferAttrKey_Align, &tempAlignment, sizeof(tempAlignment)},
        {NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess)},
        {NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
        {NvSciBufGeneralAttrKey_GpuId, &m_devUUID, sizeof(m_devUUID)},
    };

    checkNvSciErrors(NvSciBufAttrListSetAttrs(
        m_rawBufAttrList, rawBufAttrs,
        sizeof(rawBufAttrs) / sizeof(NvSciBufAttrKeyValuePair)));
  }

  NvSciSyncAttrList getNvSciSyncAttrList() { return m_syncAttrList; }

  NvSciBufAttrList getNvSciRawBufAttrList() { return m_rawBufAttrList; }

  void runImageGrayscale(std::string image_filename, size_t imageWidth,
                         size_t imageHeight) {
    int numOfGPUs = 0;
    checkCudaErrors(cudaGetDeviceCount(&numOfGPUs));  // For cuda init purpose
    checkCudaErrors(cudaSetDevice(m_cudaDeviceId));

    waitExternalSemaphore();
    launchGrayScaleKernel((unsigned int *)d_outputBuf, image_filename,
                          imageWidth, imageHeight, streamToRun);
  }

  void cudaImportNvSciSemaphore(NvSciSyncObj syncObj) {
    checkCudaErrors(cudaSetDevice(m_cudaDeviceId));

    cudaExternalSemaphoreHandleDesc extSemDesc;
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDesc.handle.nvSciSyncObj = (void *)syncObj;

    checkCudaErrors(cudaImportExternalSemaphore(&waitSem, &extSemDesc));
  }

  void waitExternalSemaphore() {
    checkCudaErrors(cudaSetDevice(m_cudaDeviceId));

    cudaExternalSemaphoreWaitParams waitParams;
    memset(&waitParams, 0, sizeof(waitParams));
    // For cross-process signaler-waiter applications need to use NvSciIpc
    // and NvSciSync[Export|Import] utilities to share the NvSciSyncFence
    // across process. This step is optional in single-process.
    waitParams.params.nvSciSync.fence = (void *)m_fence;
    waitParams.flags = 0;

    checkCudaErrors(
        cudaWaitExternalSemaphoresAsync(&waitSem, &waitParams, 1, streamToRun));
  }

  void cudaImportNvSciRawBuf(NvSciBufObj inputBufObj) {
    checkCudaErrors(cudaSetDevice(m_cudaDeviceId));

    checkNvSciErrors(NvSciBufObjGetAttrList(inputBufObj, &m_buffAttrListOut));

    memset(pairArrayOut, 0, sizeof(NvSciBufAttrKeyValuePair) * 10);
    pairArrayOut[0].key = NvSciBufRawBufferAttrKey_Size;
    pairArrayOut[1].key = NvSciBufRawBufferAttrKey_Align;

    checkNvSciErrors(
        NvSciBufAttrListGetAttrs(m_buffAttrListOut, pairArrayOut, 2));

    uint64_t size = *(uint64_t *)pairArrayOut[0].value;
    uint64_t offset = *(uint64_t *)pairArrayOut[1].value;

    cudaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = inputBufObj;
    memHandleDesc.size = size;
    checkCudaErrors(cudaImportExternalMemory(&extMemRawBuf, &memHandleDesc));

    cudaExternalMemoryBufferDesc bufferDesc;
    memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.offset = offset;
    bufferDesc.size = size;
    m_bufSize = size;

    checkCudaErrors(cudaExternalMemoryGetMappedBuffer(
        &d_outputBuf, extMemRawBuf, &bufferDesc));
  }
};

void thread_rotateAndSignal(cudaNvSciSignal *cudaNvSciSignalObj,
                            unsigned char *imageData) {
  std::lock_guard<std::mutex> guard(m_mutex);
  cudaNvSciSignalObj->runRotateImageAndSignal(imageData);
  workSubmitted = true;
  m_condVar.notify_one();
}

void thread_waitAndGrayscale(cudaNvSciWait *cudaNvSciWaitObj,
                             std::string image_filename, size_t imageWidth,
                             size_t imageHeight) {
  // Acquire the lock
  std::unique_lock<std::mutex> mlock(m_mutex);
  m_condVar.wait(mlock, [] { return workSubmitted; });
  cudaNvSciWaitObj->runImageGrayscale(image_filename, imageWidth, imageHeight);
}

cudaNvSci::cudaNvSci(int isMultiGPU, std::vector<int> &deviceIds,
                     unsigned char *imageData, size_t width, size_t height)
    : m_isMultiGPU(isMultiGPU),
      image_data(imageData),
      imageWidth(width),
      imageHeight(height) {
  if (isMultiGPU) {
    m_cudaNvSciSignalDeviceId = deviceIds[0];
    m_cudaNvSciWaitDeviceId = deviceIds[1];
  } else {
    m_cudaNvSciSignalDeviceId = m_cudaNvSciWaitDeviceId = deviceIds[0];
  }

  m_bufSize = imageWidth * imageHeight * sizeof(unsigned int);
}

void cudaNvSci::initNvSci() {
  checkNvSciErrors(NvSciSyncModuleOpen(&syncModule));
  checkNvSciErrors(NvSciBufModuleOpen(&buffModule));
  fence = (NvSciSyncFence *)calloc(1, sizeof(NvSciSyncFence));
}

void cudaNvSci::runCudaNvSci(std::string &image_filename) {
  initNvSci();

  cudaNvSciSignal rotateAndSignal(buffModule, syncModule,
                                  m_cudaNvSciSignalDeviceId, m_bufSize,
                                  imageWidth, imageHeight, fence);
  cudaNvSciWait waitAndGrayscale(buffModule, syncModule,
                                 m_cudaNvSciWaitDeviceId, m_bufSize, fence);

  rawBufUnreconciledList[0] = rotateAndSignal.getNvSciRawBufAttrList();
  rawBufUnreconciledList[1] = waitAndGrayscale.getNvSciRawBufAttrList();

  createNvSciRawBufObj();

  imageBufUnreconciledList[0] = rotateAndSignal.getNvSciImageBufAttrList();

  createNvSciBufImageObj();

  rotateAndSignal.cudaImportNvSciRawBuf(rawBufObj);
  rotateAndSignal.cudaImportNvSciImage(imageBufObj);

  waitAndGrayscale.cudaImportNvSciRawBuf(rawBufObj);

  syncUnreconciledList[0] = rotateAndSignal.getNvSciSyncAttrList();
  syncUnreconciledList[1] = waitAndGrayscale.getNvSciSyncAttrList();

  createNvSciSyncObj();

  rotateAndSignal.cudaImportNvSciSemaphore(syncObj);
  waitAndGrayscale.cudaImportNvSciSemaphore(syncObj);

  std::thread rotateThread(&thread_rotateAndSignal, &rotateAndSignal,
                           image_data);

  std::thread grayscaleThread(&thread_waitAndGrayscale, &waitAndGrayscale,
                              image_filename, imageWidth, imageHeight);

  rotateThread.join();
  grayscaleThread.join();
}

void cudaNvSci::createNvSciRawBufObj() {
  int numAttrList = 2;
  checkNvSciErrors(NvSciBufAttrListReconcile(rawBufUnreconciledList,
                                             numAttrList, &rawBufReconciledList,
                                             &buffConflictList));
  checkNvSciErrors(NvSciBufObjAlloc(rawBufReconciledList, &rawBufObj));
  printf("created NvSciBufObj\n");
}

void cudaNvSci::createNvSciBufImageObj() {
  int numAttrList = 1;
  checkNvSciErrors(NvSciBufAttrListReconcile(
      imageBufUnreconciledList, numAttrList, &imageBufReconciledList,
      &imageBufConflictList));
  checkNvSciErrors(NvSciBufObjAlloc(imageBufReconciledList, &imageBufObj));
  printf("created NvSciBufImageObj\n");
}

void cudaNvSci::createNvSciSyncObj() {
  int numAttrList = 2;
  checkNvSciErrors(NvSciSyncAttrListReconcile(syncUnreconciledList, numAttrList,
                                              &syncReconciledList,
                                              &syncConflictList));
  checkNvSciErrors(NvSciSyncObjAlloc(syncReconciledList, &syncObj));
  printf("created NvSciSyncObj\n");
}
