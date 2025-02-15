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

#include "cudla.h"
#include "nvscierror.h"
#include "nvscibuf.h"
#include "nvscisync.h"

#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <unistd.h>

#define DPRINTF(...) printf(__VA_ARGS__)

static void printTensorDesc(cudlaModuleTensorDescriptor* tensorDesc) {
  DPRINTF("\tTENSOR NAME : %s\n", tensorDesc->name);
  DPRINTF("\tsize: %lu\n", tensorDesc->size);

  DPRINTF("\tdims: [%lu, %lu, %lu, %lu]\n", tensorDesc->n, tensorDesc->c,
          tensorDesc->h, tensorDesc->w);

  DPRINTF("\tdata fmt: %d\n", tensorDesc->dataFormat);
  DPRINTF("\tdata type: %d\n", tensorDesc->dataType);
  DPRINTF("\tdata category: %d\n", tensorDesc->dataCategory);
  DPRINTF("\tpixel fmt: %d\n", tensorDesc->pixelFormat);
  DPRINTF("\tpixel mapping: %d\n", tensorDesc->pixelMapping);
  DPRINTF("\tstride[0]: %d\n", tensorDesc->stride[0]);
  DPRINTF("\tstride[1]: %d\n", tensorDesc->stride[1]);
  DPRINTF("\tstride[2]: %d\n", tensorDesc->stride[2]);
  DPRINTF("\tstride[3]: %d\n", tensorDesc->stride[3]);
}

static int initializeInputBuffers(char* filePath,
                                  cudlaModuleTensorDescriptor* tensorDesc,
                                  unsigned char* buf) {
  // Read the file in filePath and fill up 'buf' according to format
  // specified by the user.

  return 0;
}

typedef struct {
  cudlaDevHandle devHandle;
  cudlaModule moduleHandle;
  unsigned char* loadableData;
  unsigned char* inputBuffer;
  unsigned char* outputBuffer;
  NvSciBufObj inputBufObj;
  NvSciBufObj outputBufObj;
  NvSciBufModule bufModule;
  NvSciBufAttrList inputAttrList;
  NvSciBufAttrList reconciledInputAttrList;
  NvSciBufAttrList inputConflictList;
  NvSciBufAttrList outputAttrList;
  NvSciBufAttrList reconciledOutputAttrList;
  NvSciBufAttrList outputConflictList;
  NvSciSyncObj syncObj1;
  NvSciSyncObj syncObj2;
  NvSciSyncModule syncModule;
  NvSciSyncFence preFence;
  NvSciSyncFence eofFence;
  NvSciSyncCpuWaitContext nvSciCtx;
  NvSciSyncAttrList waiterAttrListObj1;
  NvSciSyncAttrList signalerAttrListObj1;
  NvSciSyncAttrList waiterAttrListObj2;
  NvSciSyncAttrList signalerAttrListObj2;
  NvSciSyncAttrList nvSciSyncConflictListObj1;
  NvSciSyncAttrList nvSciSyncReconciledListObj1;
  NvSciSyncAttrList nvSciSyncConflictListObj2;
  NvSciSyncAttrList nvSciSyncReconciledListObj2;
  cudlaModuleTensorDescriptor* inputTensorDesc;
  cudlaModuleTensorDescriptor* outputTensorDesc;
  CudlaFence* preFences;
  uint64_t** devPtrs;
  cudlaWaitEvents* waitEvents;
  cudlaSignalEvents* signalEvents;
} ResourceList;

void cleanUp(ResourceList* resourceList);

void cleanUp(ResourceList* resourceList) {
  if (resourceList->inputBufObj != NULL) {
    NvSciBufObjFree(resourceList->inputBufObj);
    resourceList->inputBufObj = NULL;
  }

  if (resourceList->outputBufObj != NULL) {
    NvSciBufObjFree(resourceList->outputBufObj);
    resourceList->outputBufObj = NULL;
  }

  if (resourceList->reconciledInputAttrList != NULL) {
    NvSciBufAttrListFree(resourceList->reconciledInputAttrList);
    resourceList->reconciledInputAttrList = NULL;
  }

  if (resourceList->inputConflictList != NULL) {
    NvSciBufAttrListFree(resourceList->inputConflictList);
    resourceList->inputConflictList = NULL;
  }

  if (resourceList->inputAttrList != NULL) {
    NvSciBufAttrListFree(resourceList->inputAttrList);
    resourceList->inputAttrList = NULL;
  }

  if (resourceList->reconciledOutputAttrList != NULL) {
    NvSciBufAttrListFree(resourceList->reconciledOutputAttrList);
    resourceList->reconciledOutputAttrList = NULL;
  }

  if (resourceList->outputConflictList != NULL) {
    NvSciBufAttrListFree(resourceList->outputConflictList);
    resourceList->outputConflictList = NULL;
  }

  if (resourceList->outputAttrList != NULL) {
    NvSciBufAttrListFree(resourceList->outputAttrList);
    resourceList->outputAttrList = NULL;
  }

  if (resourceList->bufModule != NULL) {
    NvSciBufModuleClose(resourceList->bufModule);
    resourceList->bufModule = NULL;
  }

  NvSciSyncFenceClear(&(resourceList->preFence));
  NvSciSyncFenceClear(&(resourceList->eofFence));

  if (resourceList->syncObj1 != NULL) {
    NvSciSyncObjFree(resourceList->syncObj1);
    resourceList->syncObj1 = NULL;
  }

  if (resourceList->syncObj2 != NULL) {
    NvSciSyncObjFree(resourceList->syncObj2);
    resourceList->syncObj2 = NULL;
  }

  if (resourceList->nvSciSyncConflictListObj1 != NULL) {
    NvSciSyncAttrListFree(resourceList->nvSciSyncConflictListObj1);
    resourceList->nvSciSyncConflictListObj1 = NULL;
  }

  if (resourceList->nvSciSyncReconciledListObj1 != NULL) {
    NvSciSyncAttrListFree(resourceList->nvSciSyncReconciledListObj1);
    resourceList->nvSciSyncReconciledListObj1 = NULL;
  }

  if (resourceList->nvSciSyncConflictListObj2 != NULL) {
    NvSciSyncAttrListFree(resourceList->nvSciSyncConflictListObj2);
    resourceList->nvSciSyncConflictListObj2 = NULL;
  }

  if (resourceList->nvSciSyncReconciledListObj2 != NULL) {
    NvSciSyncAttrListFree(resourceList->nvSciSyncReconciledListObj2);
    resourceList->nvSciSyncReconciledListObj2 = NULL;
  }

  if (resourceList->signalerAttrListObj1 != NULL) {
    NvSciSyncAttrListFree(resourceList->signalerAttrListObj1);
    resourceList->signalerAttrListObj1 = NULL;
  }

  if (resourceList->waiterAttrListObj1 != NULL) {
    NvSciSyncAttrListFree(resourceList->waiterAttrListObj1);
    resourceList->waiterAttrListObj1 = NULL;
  }

  if (resourceList->signalerAttrListObj2 != NULL) {
    NvSciSyncAttrListFree(resourceList->signalerAttrListObj2);
    resourceList->signalerAttrListObj2 = NULL;
  }

  if (resourceList->waiterAttrListObj2 != NULL) {
    NvSciSyncAttrListFree(resourceList->waiterAttrListObj2);
    resourceList->waiterAttrListObj2 = NULL;
  }

  if (resourceList->nvSciCtx != NULL) {
    NvSciSyncCpuWaitContextFree(resourceList->nvSciCtx);
    resourceList->nvSciCtx = NULL;
  }

  if (resourceList->syncModule != NULL) {
    NvSciSyncModuleClose(resourceList->syncModule);
    resourceList->syncModule = NULL;
  }

  if (resourceList->waitEvents != NULL) {
    free(resourceList->waitEvents);
    resourceList->waitEvents = NULL;
  }

  if (resourceList->preFences != NULL) {
    free(resourceList->preFences);
    resourceList->preFences = NULL;
  }

  if (resourceList->signalEvents != NULL) {
    if (resourceList->signalEvents->eofFences != NULL) {
      free(resourceList->signalEvents->eofFences);
      resourceList->signalEvents->eofFences = NULL;
    }

    free(resourceList->signalEvents);
    resourceList->signalEvents = NULL;
  }

  if (resourceList->devPtrs != NULL) {
    free(resourceList->devPtrs);
    resourceList->devPtrs = NULL;
  }

  if (resourceList->inputTensorDesc != NULL) {
    free(resourceList->inputTensorDesc);
    resourceList->inputTensorDesc = NULL;
  }
  if (resourceList->outputTensorDesc != NULL) {
    free(resourceList->outputTensorDesc);
    resourceList->outputTensorDesc = NULL;
  }

  if (resourceList->loadableData != NULL) {
    free(resourceList->loadableData);
    resourceList->loadableData = NULL;
  }

  if (resourceList->moduleHandle != NULL) {
    cudlaModuleUnload(resourceList->moduleHandle, 0);
    resourceList->moduleHandle = NULL;
  }

  if (resourceList->devHandle != NULL) {
    cudlaDestroyDevice(resourceList->devHandle);
    resourceList->devHandle = NULL;
  }

  if (resourceList->inputBuffer != NULL) {
    free(resourceList->inputBuffer);
    resourceList->inputBuffer = NULL;
  }
  if (resourceList->outputBuffer != NULL) {
    free(resourceList->outputBuffer);
    resourceList->outputBuffer = NULL;
  }
}

cudlaStatus createAndSetAttrList(NvSciBufModule module, uint64_t bufSize,
                                 NvSciBufAttrList* attrList);

cudlaStatus createAndSetAttrList(NvSciBufModule module, uint64_t bufSize,
                                 NvSciBufAttrList* attrList) {
  cudlaStatus status = cudlaSuccess;
  NvSciError sciStatus = NvSciError_Success;

  sciStatus = NvSciBufAttrListCreate(module, attrList);
  if (sciStatus != NvSciError_Success) {
    status = cudlaErrorNvSci;
    DPRINTF("Error in creating NvSciBuf attribute list\n");
    return status;
  }

  bool needCpuAccess = true;
  NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
  uint32_t dimcount = 1;
  uint64_t sizes[] = {bufSize};
  uint32_t alignment[] = {1};
  uint32_t dataType = NvSciDataType_Int8;
  NvSciBufType type = NvSciBufType_Tensor;
  uint64_t baseAddrAlign = 512;

  NvSciBufAttrKeyValuePair setAttrs[] = {
      {.key = NvSciBufGeneralAttrKey_Types,
       .value = &type,
       .len = sizeof(type)},
      {.key = NvSciBufTensorAttrKey_DataType,
       .value = &dataType,
       .len = sizeof(dataType)},
      {.key = NvSciBufTensorAttrKey_NumDims,
       .value = &dimcount,
       .len = sizeof(dimcount)},
      {.key = NvSciBufTensorAttrKey_SizePerDim,
       .value = &sizes,
       .len = sizeof(sizes)},
      {.key = NvSciBufTensorAttrKey_AlignmentPerDim,
       .value = &alignment,
       .len = sizeof(alignment)},
      {.key = NvSciBufTensorAttrKey_BaseAddrAlign,
       .value = &baseAddrAlign,
       .len = sizeof(baseAddrAlign)},
      {.key = NvSciBufGeneralAttrKey_RequiredPerm,
       .value = &perm,
       .len = sizeof(perm)},
      {.key = NvSciBufGeneralAttrKey_NeedCpuAccess,
       .value = &needCpuAccess,
       .len = sizeof(needCpuAccess)}};
  size_t length = sizeof(setAttrs) / sizeof(NvSciBufAttrKeyValuePair);

  sciStatus = NvSciBufAttrListSetAttrs(*attrList, setAttrs, length);
  if (sciStatus != NvSciError_Success) {
    status = cudlaErrorNvSci;
    DPRINTF("Error in setting NvSciBuf attribute list\n");
    return status;
  }

  return status;
}

NvSciError fillCpuSignalerAttrList(NvSciSyncAttrList list);

NvSciError fillCpuSignalerAttrList(NvSciSyncAttrList list) {
  bool cpuSignaler = true;
  NvSciSyncAttrKeyValuePair keyValue[2];
  memset(keyValue, 0, sizeof(keyValue));
  keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
  keyValue[0].value = (void*)&cpuSignaler;
  keyValue[0].len = sizeof(cpuSignaler);

  NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_SignalOnly;
  keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
  keyValue[1].value = (void*)&cpuPerm;
  keyValue[1].len = sizeof(cpuPerm);

  return NvSciSyncAttrListSetAttrs(list, keyValue, 2);
}

NvSciError fillCpuWaiterAttrList(NvSciSyncAttrList list);

NvSciError fillCpuWaiterAttrList(NvSciSyncAttrList list) {
  bool cpuWaiter = true;
  NvSciSyncAttrKeyValuePair keyValue[2];
  memset(keyValue, 0, sizeof(keyValue));
  keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
  keyValue[0].value = (void*)&cpuWaiter;
  keyValue[0].len = sizeof(cpuWaiter);

  NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
  keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
  keyValue[1].value = (void*)&cpuPerm;
  keyValue[1].len = sizeof(cpuPerm);

  return NvSciSyncAttrListSetAttrs(list, keyValue, 2);
}

int main(int argc, char** argv) {
  cudlaDevHandle devHandle;
  cudlaModule moduleHandle;
  cudlaStatus err;
  FILE* fp = NULL;
  struct stat st;
  size_t file_size;
  size_t actually_read = 0;
  unsigned char* loadableData = NULL;

  ResourceList resourceList;

  memset(&resourceList, 0x00, sizeof(ResourceList));

  if (argc != 3) {
    DPRINTF("Usage : ./cuDLAStandaloneMode <loadable> <imageFile>\n");
    return 1;
  }

  // Read loadable into buffer.
  fp = fopen(argv[1], "rb");
  if (fp == NULL) {
    DPRINTF("Cannot open file %s\n", argv[1]);
    return 1;
  }

  if (stat(argv[1], &st) != 0) {
    DPRINTF("Cannot stat file\n");
    return 1;
  }

  file_size = st.st_size;
  DPRINTF("The file size = %ld\n", file_size);

  loadableData = (unsigned char*)malloc(file_size);
  if (loadableData == NULL) {
    DPRINTF("Cannot Allocate memory for loadable\n");
    return 1;
  }

  actually_read = fread(loadableData, 1, file_size, fp);
  if (actually_read != file_size) {
    free(loadableData);
    DPRINTF("Read wrong size\n");
    return 1;
  }
  fclose(fp);

  resourceList.loadableData = loadableData;

  err = cudlaCreateDevice(0, &devHandle, CUDLA_STANDALONE);
  if (err != cudlaSuccess) {
    DPRINTF("Error in cuDLA create device = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  DPRINTF("Device created successfully\n");
  resourceList.devHandle = devHandle;

  err = cudlaModuleLoadFromMemory(devHandle, loadableData, file_size,
                                  &moduleHandle, 0);
  if (err != cudlaSuccess) {
    DPRINTF("Error in cudlaModuleLoadFromMemory = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  } else {
    DPRINTF("Successfully loaded module\n");
  }

  resourceList.moduleHandle = moduleHandle;
  // Get tensor attributes.
  uint32_t numInputTensors = 0;
  uint32_t numOutputTensors = 0;
  cudlaModuleAttribute attribute;

  err = cudlaModuleGetAttributes(moduleHandle, CUDLA_NUM_INPUT_TENSORS,
                                 &attribute);
  if (err != cudlaSuccess) {
    DPRINTF("Error in getting numInputTensors = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }
  numInputTensors = attribute.numInputTensors;
  DPRINTF("numInputTensors = %d\n", numInputTensors);

  err = cudlaModuleGetAttributes(moduleHandle, CUDLA_NUM_OUTPUT_TENSORS,
                                 &attribute);
  if (err != cudlaSuccess) {
    DPRINTF("Error in getting numOutputTensors = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }
  numOutputTensors = attribute.numOutputTensors;
  DPRINTF("numOutputTensors = %d\n", numOutputTensors);

  cudlaModuleTensorDescriptor* inputTensorDesc =
      (cudlaModuleTensorDescriptor*)malloc(sizeof(cudlaModuleTensorDescriptor) *
                                           numInputTensors);
  cudlaModuleTensorDescriptor* outputTensorDesc =
      (cudlaModuleTensorDescriptor*)malloc(sizeof(cudlaModuleTensorDescriptor) *
                                           numOutputTensors);

  if ((inputTensorDesc == NULL) || (outputTensorDesc == NULL)) {
    if (inputTensorDesc != NULL) {
      free(inputTensorDesc);
      inputTensorDesc = NULL;
    }

    if (outputTensorDesc != NULL) {
      free(outputTensorDesc);
      outputTensorDesc = NULL;
    }

    cleanUp(&resourceList);
    return 1;
  }

  resourceList.inputTensorDesc = inputTensorDesc;
  resourceList.outputTensorDesc = outputTensorDesc;

  attribute.inputTensorDesc = inputTensorDesc;
  err = cudlaModuleGetAttributes(moduleHandle, CUDLA_INPUT_TENSOR_DESCRIPTORS,
                                 &attribute);
  if (err != cudlaSuccess) {
    DPRINTF("Error in getting input tensor descriptor = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }
  DPRINTF("Printing input tensor descriptor\n");
  printTensorDesc(inputTensorDesc);

  attribute.outputTensorDesc = outputTensorDesc;
  err = cudlaModuleGetAttributes(moduleHandle, CUDLA_OUTPUT_TENSOR_DESCRIPTORS,
                                 &attribute);
  if (err != cudlaSuccess) {
    DPRINTF("Error in getting output tensor descriptor = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }
  DPRINTF("Printing output tensor descriptor\n");
  printTensorDesc(outputTensorDesc);

  // Setup the input and output buffers which will be used as an input to CUDA.
  unsigned char* inputBuffer = (unsigned char*)malloc(inputTensorDesc[0].size);
  if (inputBuffer == NULL) {
    DPRINTF("Error in allocating input memory\n");
    cleanUp(&resourceList);
    return 1;
  }

  resourceList.inputBuffer = inputBuffer;

  unsigned char* outputBuffer =
      (unsigned char*)malloc(outputTensorDesc[0].size);
  if (outputBuffer == NULL) {
    DPRINTF("Error in allocating output memory\n");
    cleanUp(&resourceList);
    return 1;
  }

  resourceList.outputBuffer = outputBuffer;

  memset(inputBuffer, 0x00, inputTensorDesc[0].size);
  memset(outputBuffer, 0x00, outputTensorDesc[0].size);

  // Fill up the buffers with data.
  if (initializeInputBuffers(argv[2], inputTensorDesc, inputBuffer) != 0) {
    DPRINTF("Error in initializing input buffer from PGM image\n");
    cleanUp(&resourceList);
    return 1;
  }

  NvSciBufModule bufModule = NULL;
  NvSciBufAttrList inputAttrList = NULL;
  NvSciBufAttrList outputAttrList = NULL;
  NvSciBufAttrList reconciledInputAttrList = NULL;
  NvSciBufAttrList reconciledOutputAttrList = NULL;
  NvSciBufAttrList inputConflictList = NULL;
  NvSciBufAttrList outputConflictList = NULL;
  NvSciError sciError = NvSciError_Success;

  sciError = NvSciBufModuleOpen(&bufModule);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in initializing NvSciBufModule\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.bufModule = bufModule;

  // creating and setting input attribute list
  err =
      createAndSetAttrList(bufModule, inputTensorDesc[0].size, &inputAttrList);
  if (err != cudlaSuccess) {
    DPRINTF("Error in creating NvSciBuf attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.inputAttrList = inputAttrList;

  sciError = NvSciBufAttrListReconcile(
      &inputAttrList, 1, &reconciledInputAttrList, &inputConflictList);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in reconciling NvSciBuf attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }

  resourceList.reconciledInputAttrList = reconciledInputAttrList;
  resourceList.inputConflictList = inputConflictList;

  // creating and setting output attribute list
  err = createAndSetAttrList(bufModule, outputTensorDesc[0].size,
                             &outputAttrList);
  if (err != cudlaSuccess) {
    DPRINTF("Error in creating NvSciBuf attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.outputAttrList = outputAttrList;

  sciError = NvSciBufAttrListReconcile(
      &outputAttrList, 1, &reconciledOutputAttrList, &outputConflictList);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in reconciling NvSciBuf attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }

  resourceList.reconciledOutputAttrList = reconciledOutputAttrList;
  resourceList.outputConflictList = outputConflictList;

  NvSciBufObj inputBufObj, outputBufObj;
  sciError = NvSciBufObjAlloc(reconciledInputAttrList, &inputBufObj);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in allocating NvSciBuf object\n");
    cleanUp(&resourceList);
    return 1;
  }

  resourceList.inputBufObj = inputBufObj;

  sciError = NvSciBufObjAlloc(reconciledOutputAttrList, &outputBufObj);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in allocating NvSciBuf object\n");
    cleanUp(&resourceList);
    return 1;
  }

  resourceList.outputBufObj = outputBufObj;

  uint64_t* inputBufObjRegPtr = NULL;
  uint64_t* outputBufObjRegPtr = NULL;
  void* inputBufObjBuffer;
  void* outputBufObjBuffer;

  // importing external memory
  cudlaExternalMemoryHandleDesc memDesc = {0};
  memset(&memDesc, 0, sizeof(memDesc));
  memDesc.extBufObject = (void*)inputBufObj;
  memDesc.size = inputTensorDesc[0].size;
  err = cudlaImportExternalMemory(devHandle, &memDesc, &inputBufObjRegPtr, 0);
  if (err != cudlaSuccess) {
    DPRINTF("Error in importing external memory = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  sciError = NvSciBufObjGetCpuPtr(inputBufObj, &inputBufObjBuffer);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in getting NvSciBuf CPU pointer\n");
    cleanUp(&resourceList);
    return 1;
  }
  memcpy(inputBufObjBuffer, inputBuffer, inputTensorDesc[0].size);

  memset(&memDesc, 0, sizeof(memDesc));
  memDesc.extBufObject = (void*)outputBufObj;
  memDesc.size = outputTensorDesc[0].size;
  err = cudlaImportExternalMemory(devHandle, &memDesc, &outputBufObjRegPtr, 0);
  if (err != cudlaSuccess) {
    DPRINTF("Error in importing external memory = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  sciError = NvSciBufObjGetCpuPtr(outputBufObj, &outputBufObjBuffer);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in getting NvSciBuf CPU pointer\n");
    cleanUp(&resourceList);
    return 1;
  }
  memset(outputBufObjBuffer, 0, outputTensorDesc[0].size);

  NvSciSyncObj syncObj1, syncObj2;
  NvSciSyncModule syncModule;
  NvSciSyncAttrList syncAttrListObj1[2];
  NvSciSyncAttrList syncAttrListObj2[2];
  NvSciSyncCpuWaitContext nvSciCtx;
  NvSciSyncAttrList waiterAttrListObj1 = NULL;
  NvSciSyncAttrList signalerAttrListObj1 = NULL;
  NvSciSyncAttrList waiterAttrListObj2 = NULL;
  NvSciSyncAttrList signalerAttrListObj2 = NULL;
  NvSciSyncAttrList nvSciSyncConflictListObj1;
  NvSciSyncAttrList nvSciSyncReconciledListObj1;
  NvSciSyncAttrList nvSciSyncConflictListObj2;
  NvSciSyncAttrList nvSciSyncReconciledListObj2;

  sciError = NvSciSyncModuleOpen(&syncModule);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in initializing NvSciSyncModuleOpen\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.syncModule = syncModule;

  sciError = NvSciSyncAttrListCreate(syncModule, &signalerAttrListObj1);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in creating NvSciSync attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.signalerAttrListObj1 = signalerAttrListObj1;

  sciError = NvSciSyncAttrListCreate(syncModule, &waiterAttrListObj1);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in creating NvSciSync attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.waiterAttrListObj1 = waiterAttrListObj1;

  err = cudlaGetNvSciSyncAttributes(
      reinterpret_cast<uint64_t*>(waiterAttrListObj1),
      CUDLA_NVSCISYNC_ATTR_WAIT);
  if (err != cudlaSuccess) {
    DPRINTF("Error in getting cuDLA's NvSciSync attributes\n");
    cleanUp(&resourceList);
    return 1;
  }

  sciError = fillCpuSignalerAttrList(signalerAttrListObj1);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in setting NvSciSync attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }

  syncAttrListObj1[0] = signalerAttrListObj1;
  syncAttrListObj1[1] = waiterAttrListObj1;
  sciError = NvSciSyncAttrListReconcile(syncAttrListObj1, 2,
                                        &nvSciSyncReconciledListObj1,
                                        &nvSciSyncConflictListObj1);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in reconciling NvSciSync's attribute lists\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.nvSciSyncConflictListObj1 = nvSciSyncConflictListObj1;
  resourceList.nvSciSyncReconciledListObj1 = nvSciSyncReconciledListObj1;

  sciError = NvSciSyncObjAlloc(nvSciSyncReconciledListObj1, &syncObj1);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in allocating NvSciSync object\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.syncObj1 = syncObj1;

  sciError = NvSciSyncCpuWaitContextAlloc(syncModule, &nvSciCtx);
  if (sciError != NvSciError_Success) {
    DPRINTF(
        "Error in allocating cpu wait context NvSciSyncCpuWaitContextAlloc\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.nvSciCtx = nvSciCtx;

  sciError = NvSciSyncAttrListCreate(syncModule, &signalerAttrListObj2);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in creating NvSciSync attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.signalerAttrListObj2 = signalerAttrListObj2;

  sciError = NvSciSyncAttrListCreate(syncModule, &waiterAttrListObj2);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in creating NvSciSync attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.waiterAttrListObj2 = waiterAttrListObj2;

  err = cudlaGetNvSciSyncAttributes(
      reinterpret_cast<uint64_t*>(signalerAttrListObj2),
      CUDLA_NVSCISYNC_ATTR_SIGNAL);
  if (err != cudlaSuccess) {
    DPRINTF("Error in getting cuDLA's NvSciSync attributes\n");
    cleanUp(&resourceList);
    return 1;
  }

  sciError = fillCpuWaiterAttrList(waiterAttrListObj2);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in setting NvSciSync attribute list\n");
    cleanUp(&resourceList);
    return 1;
  }

  syncAttrListObj2[0] = signalerAttrListObj2;
  syncAttrListObj2[1] = waiterAttrListObj2;
  sciError = NvSciSyncAttrListReconcile(syncAttrListObj2, 2,
                                        &nvSciSyncReconciledListObj2,
                                        &nvSciSyncConflictListObj2);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in reconciling NvSciSync's attribute lists\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.nvSciSyncConflictListObj2 = nvSciSyncConflictListObj2;
  resourceList.nvSciSyncReconciledListObj2 = nvSciSyncReconciledListObj2;

  sciError = NvSciSyncObjAlloc(nvSciSyncReconciledListObj2, &syncObj2);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in allocating NvSciSync object\n");
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.syncObj2 = syncObj2;

  // importing external semaphore
  uint64_t* nvSciSyncObjRegPtr1 = NULL;
  uint64_t* nvSciSyncObjRegPtr2 = NULL;
  cudlaExternalSemaphoreHandleDesc semaMemDesc = {0};
  memset(&semaMemDesc, 0, sizeof(semaMemDesc));
  semaMemDesc.extSyncObject = syncObj1;
  err = cudlaImportExternalSemaphore(devHandle, &semaMemDesc,
                                     &nvSciSyncObjRegPtr1, 0);
  if (err != cudlaSuccess) {
    DPRINTF("Error in importing external semaphore = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  memset(&semaMemDesc, 0, sizeof(semaMemDesc));
  semaMemDesc.extSyncObject = syncObj2;
  err = cudlaImportExternalSemaphore(devHandle, &semaMemDesc,
                                     &nvSciSyncObjRegPtr2, 0);
  if (err != cudlaSuccess) {
    DPRINTF("Error in importing external semaphore = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  DPRINTF("ALL MEMORY REGISTERED SUCCESSFULLY\n");

  // Wait events
  NvSciSyncFence preFence = NvSciSyncFenceInitializer;
  sciError = NvSciSyncObjGenerateFence(syncObj1, &preFence);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in generating NvSciSyncObj fence %x\n", sciError);
    cleanUp(&resourceList);
    return 1;
  }
  resourceList.preFence = preFence;

  cudlaWaitEvents* waitEvents;
  waitEvents = (cudlaWaitEvents*)malloc(sizeof(cudlaWaitEvents));
  if (waitEvents == NULL) {
    DPRINTF("Error in allocating wait events\n");
    cleanUp(&resourceList);
    return 1;
  }

  waitEvents->numEvents = 1;
  CudlaFence* preFences =
      (CudlaFence*)malloc(waitEvents->numEvents * sizeof(CudlaFence));
  if (preFences == NULL) {
    DPRINTF("Error in allocating preFence array\n");
    cleanUp(&resourceList);
    return 1;
  }

  preFences[0].fence = &preFence;
  preFences[0].type = CUDLA_NVSCISYNC_FENCE;
  waitEvents->preFences = preFences;
  resourceList.preFences = preFences;
  resourceList.waitEvents = waitEvents;

  // Signal Events
  cudlaSignalEvents* signalEvents;
  signalEvents = (cudlaSignalEvents*)malloc(sizeof(cudlaSignalEvents));
  if (signalEvents == NULL) {
    DPRINTF("Error in allocating signal events\n");
    cleanUp(&resourceList);
    return 1;
  }

  signalEvents->numEvents = 1;
  uint64_t** devPtrs =
      (uint64_t**)malloc(signalEvents->numEvents * sizeof(uint64_t*));
  if (devPtrs == NULL) {
    DPRINTF(
        "Error in allocating output pointer's array of registered objects\n");
    cleanUp(&resourceList);
    return 1;
  }
  devPtrs[0] = nvSciSyncObjRegPtr2;
  signalEvents->devPtrs = devPtrs;
  resourceList.devPtrs = devPtrs;

  signalEvents->eofFences =
      (CudlaFence*)malloc(signalEvents->numEvents * sizeof(CudlaFence));
  if (signalEvents->eofFences == NULL) {
    DPRINTF("Error in allocating eofFence array\n");
    cleanUp(&resourceList);
    return 1;
  }

  NvSciSyncFence eofFence = NvSciSyncFenceInitializer;
  signalEvents->eofFences[0].fence = &eofFence;
  signalEvents->eofFences[0].type = CUDLA_NVSCISYNC_FENCE;
  resourceList.signalEvents = signalEvents;
  resourceList.eofFence = eofFence;

  // Enqueue a cuDLA task.
  cudlaTask task;
  task.moduleHandle = moduleHandle;
  task.outputTensor = &outputBufObjRegPtr;
  task.numOutputTensors = 1;
  task.numInputTensors = 1;
  task.inputTensor = &inputBufObjRegPtr;
  task.waitEvents = waitEvents;
  task.signalEvents = signalEvents;
  err = cudlaSubmitTask(devHandle, &task, 1, NULL, 0);
  if (err != cudlaSuccess) {
    DPRINTF("Error in submitting task\n");
    cleanUp(&resourceList);
    return 1;
  }
  DPRINTF("SUBMIT IS DONE !!!\n");

  // Signal wait events
  NvSciSyncObjSignal(syncObj1);
  DPRINTF("SIGNALED WAIT EVENTS SUCCESSFULLY\n");

  // Wait for operations to finish and bring output buffer to CPU.
  sciError = NvSciSyncFenceWait(
      reinterpret_cast<NvSciSyncFence*>(signalEvents->eofFences[0].fence),
      nvSciCtx, -1);
  if (sciError != NvSciError_Success) {
    DPRINTF("Error in waiting on NvSciSyncFence\n");
    cleanUp(&resourceList);
    return 1;
  }

  memcpy(outputBuffer, outputBufObjBuffer, outputTensorDesc[0].size);

  // Output is available in outputBuffer.

  // Teardown.
  err = cudlaMemUnregister(devHandle, inputBufObjRegPtr);
  if (err != cudlaSuccess) {
    DPRINTF("Error in unregistering external memory = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  err = cudlaMemUnregister(devHandle, outputBufObjRegPtr);
  if (err != cudlaSuccess) {
    DPRINTF("Error in unregistering external memory = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  err = cudlaMemUnregister(devHandle, nvSciSyncObjRegPtr1);
  if (err != cudlaSuccess) {
    DPRINTF("Error in unregistering external semaphore = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  err = cudlaMemUnregister(devHandle, nvSciSyncObjRegPtr2);
  if (err != cudlaSuccess) {
    DPRINTF("Error in unregistering external semaphore = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  DPRINTF("ALL MEMORY UNREGISTERED SUCCESSFULLY\n");

  free(inputTensorDesc);
  free(outputTensorDesc);
  free(loadableData);
  free(inputBuffer);
  free(outputBuffer);
  NvSciBufObjFree(inputBufObj);
  NvSciBufObjFree(outputBufObj);
  NvSciBufAttrListFree(reconciledInputAttrList);
  NvSciBufAttrListFree(inputConflictList);
  NvSciBufAttrListFree(inputAttrList);
  NvSciBufAttrListFree(reconciledOutputAttrList);
  NvSciBufAttrListFree(outputConflictList);
  NvSciBufAttrListFree(outputAttrList);
  NvSciBufModuleClose(bufModule);
  NvSciSyncObjFree(syncObj1);
  NvSciSyncObjFree(syncObj2);
  NvSciSyncAttrListFree(signalerAttrListObj1);
  NvSciSyncAttrListFree(waiterAttrListObj1);
  NvSciSyncAttrListFree(signalerAttrListObj2);
  NvSciSyncAttrListFree(waiterAttrListObj2);
  NvSciSyncAttrListFree(nvSciSyncConflictListObj1);
  NvSciSyncAttrListFree(nvSciSyncReconciledListObj1);
  NvSciSyncAttrListFree(nvSciSyncConflictListObj2);
  NvSciSyncAttrListFree(nvSciSyncReconciledListObj2);
  NvSciSyncCpuWaitContextFree(nvSciCtx);
  NvSciSyncModuleClose(syncModule);
  free(waitEvents);
  free(preFences);
  free(signalEvents->eofFences);
  free(signalEvents);
  free(devPtrs);
  NvSciSyncFenceClear(&preFence);
  NvSciSyncFenceClear(&eofFence);

  resourceList.inputTensorDesc = NULL;
  resourceList.outputTensorDesc = NULL;
  resourceList.loadableData = NULL;
  resourceList.inputBuffer = NULL;
  resourceList.outputBuffer = NULL;
  resourceList.inputBufObj = NULL;
  resourceList.outputBufObj = NULL;
  resourceList.reconciledInputAttrList = NULL;
  resourceList.inputConflictList = NULL;
  resourceList.inputAttrList = NULL;
  resourceList.reconciledOutputAttrList = NULL;
  resourceList.outputConflictList = NULL;
  resourceList.outputAttrList = NULL;
  resourceList.bufModule = NULL;
  resourceList.syncObj1 = NULL;
  resourceList.syncObj2 = NULL;
  resourceList.signalerAttrListObj1 = NULL;
  resourceList.waiterAttrListObj1 = NULL;
  resourceList.signalerAttrListObj2 = NULL;
  resourceList.waiterAttrListObj2 = NULL;
  resourceList.nvSciSyncConflictListObj1 = NULL;
  resourceList.nvSciSyncReconciledListObj1 = NULL;
  resourceList.nvSciSyncConflictListObj2 = NULL;
  resourceList.nvSciSyncReconciledListObj2 = NULL;
  resourceList.nvSciCtx = NULL;
  resourceList.syncModule = NULL;
  resourceList.waitEvents = NULL;
  resourceList.signalEvents = NULL;
  resourceList.preFences = NULL;
  resourceList.devPtrs = NULL;

  err = cudlaModuleUnload(moduleHandle, 0);
  if (err != cudlaSuccess) {
    DPRINTF("Error in cudlaModuleUnload = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  } else {
    DPRINTF("Successfully unloaded module\n");
  }

  resourceList.moduleHandle = NULL;

  err = cudlaDestroyDevice(devHandle);
  if (err != cudlaSuccess) {
    DPRINTF("Error in cuDLA destroy device = %d\n", err);
    return 1;
  }
  DPRINTF("Device destroyed successfully\n");

  resourceList.devHandle = NULL;

  DPRINTF("cuDLAStandaloneMode DONE !!!\n");

  return 0;
}
