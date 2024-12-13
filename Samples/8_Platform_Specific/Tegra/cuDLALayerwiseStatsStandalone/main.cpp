/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#include "cudlaExternalEtbl.hpp"

#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <fstream>
#include <sstream>

#define MAX_FILENAME_LEN 200
#define RESERVED_SUFFIX_LEN 10

#define DPRINTF(...) printf(__VA_ARGS__)

static void printTensorDesc(cudlaModuleTensorDescriptor* tensorDesc) {
    DPRINTF("\tTENSOR NAME : %s\n", tensorDesc->name);
    DPRINTF("\tsize: %lu\n", tensorDesc->size);

    DPRINTF("\tdims: [%lu, %lu, %lu, %lu]\n",
                    tensorDesc->n,
                    tensorDesc->c,
                    tensorDesc->h,
                    tensorDesc->w);

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

typedef struct {
    cudlaDevHandle devHandle;
    cudlaModule moduleHandle;
    unsigned char* loadableData;
    uint32_t numInputTensors;
    uint32_t numOutputTensors;
    uint32_t numOutputTaskStatistics;
    unsigned char** inputBuffer;
    unsigned char** outputBuffer;
    unsigned char** statisticsOutputBuffer;
    cudlaModuleTensorDescriptor* inputTensorDesc;
    cudlaModuleTensorDescriptor* outputTensorDesc;
    cudlaModuleTensorDescriptor* outputTaskStatisticsDesc;
    NvSciBufObj* inputBufObj;
    NvSciBufObj* outputBufObj;
    NvSciBufObj* statisticsBufObj;
    NvSciBufModule bufModule;
    NvSciBufAttrList* inputAttrList;
    NvSciBufAttrList* reconciledInputAttrList;
    NvSciBufAttrList* inputConflictList;
    NvSciBufAttrList* outputAttrList;
    NvSciBufAttrList* reconciledOutputAttrList;
    NvSciBufAttrList* outputConflictList;
    NvSciSyncObj syncObj;
    NvSciSyncModule syncModule;
    NvSciSyncCpuWaitContext nvSciCtx;
    NvSciSyncAttrList waiterAttrListObj;
    NvSciSyncAttrList signalerAttrListObj;
    NvSciSyncAttrList nvSciSyncConflictListObj;
    NvSciSyncAttrList nvSciSyncReconciledListObj;
    NvSciBufAttrList* statisticsOutputAttrList;
    NvSciBufAttrList* reconciledStatisticsOutputAttrList;
    NvSciBufAttrList* statisticsOutputConflictList;
    uint64_t** inputBufObjRegPtr;
    uint64_t** outputBufObjRegPtr;
    uint64_t** statisticsBufObjRegPtr;
    uint64_t** devPtrs;
    cudlaSignalEvents* signalEvents;
    NvSciSyncFence eofFence;
    void **csv;
} ResourceList;

void cleanUp(ResourceList* resourceList);

void cleanUp(ResourceList* resourceList) {
    uint32_t ii = 0;

    if (resourceList->inputTensorDesc != NULL) {
        free(resourceList->inputTensorDesc);
        resourceList->inputTensorDesc = NULL;
    }
    if (resourceList->outputTensorDesc != NULL) {
        free(resourceList->outputTensorDesc);
        resourceList->outputTensorDesc = NULL;
    }

    if (resourceList->outputTaskStatisticsDesc != NULL) {
        free(resourceList->outputTaskStatisticsDesc);
        resourceList->outputTaskStatisticsDesc = NULL;
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

    if (resourceList->inputBufObj != NULL) {
        for (ii = 0; ii < resourceList->numInputTensors; ii++) {
            if((resourceList->inputBufObj)[ii] != NULL) {
                NvSciBufObjFree((resourceList->inputBufObj)[ii]);
                (resourceList->inputBufObj)[ii] = NULL;
            }
        }
    }

    if (resourceList->outputBufObj != NULL) {
        for (ii = 0; ii < resourceList->numOutputTensors; ii++) {
            if((resourceList->outputBufObj)[ii] != NULL) {
                NvSciBufObjFree((resourceList->outputBufObj)[ii]);
                (resourceList->outputBufObj)[ii] = NULL;
            }
        }
    }

    if (resourceList->statisticsBufObj != NULL) {
        for (ii = 0; ii < resourceList->numOutputTaskStatistics; ii++) {
            if((resourceList->statisticsBufObj)[ii] != NULL) {
                NvSciBufObjFree((resourceList->statisticsBufObj)[ii]);
                (resourceList->statisticsBufObj)[ii] = NULL;
            }
        }
    }

    if (resourceList->inputBuffer != NULL) {
        for (ii = 0; ii < resourceList->numInputTensors; ii++) {
            if ((resourceList->inputBuffer)[ii] != NULL) {
                free((resourceList->inputBuffer)[ii]);
                (resourceList->inputBuffer)[ii] = NULL;
            }
        }
        free(resourceList->inputBuffer);
        resourceList->inputBuffer = NULL;
    }

    if (resourceList->outputBuffer != NULL) {
        for (ii = 0; ii < resourceList->numOutputTensors; ii++) {
            if ((resourceList->outputBuffer)[ii] != NULL) {
                free((resourceList->outputBuffer)[ii]);
                (resourceList->outputBuffer)[ii] = NULL;
            }
        }
        free(resourceList->outputBuffer);
        resourceList->outputBuffer = NULL;
    }

    if (resourceList->statisticsOutputBuffer != NULL) {
        for (ii = 0; ii < resourceList->numOutputTaskStatistics; ii++) {
            if ((resourceList->statisticsOutputBuffer)[ii] != NULL) {
                free((resourceList->statisticsOutputBuffer)[ii]);
                (resourceList->statisticsOutputBuffer)[ii] = NULL;
            }
        }
        free(resourceList->statisticsOutputBuffer);
        resourceList->statisticsOutputBuffer = NULL;
    }

    if (resourceList->csv != NULL) {
        for (ii = 0; ii < resourceList->numOutputTaskStatistics; ii++) {
            if ((resourceList->csv)[ii] != NULL) {
                free((resourceList->csv)[ii]);
                (resourceList->csv)[ii] = NULL;
            }
        }
        free(resourceList->csv);
        resourceList->csv = NULL;
    }

    if (resourceList->reconciledInputAttrList != NULL) {
        for (ii = 0; ii < resourceList->numInputTensors; ii++) {
            if((resourceList->reconciledInputAttrList)[ii] != NULL) {
                NvSciBufAttrListFree((resourceList->reconciledInputAttrList)[ii]);
                (resourceList->reconciledInputAttrList)[ii] = NULL;
            }
        }
        free(resourceList->reconciledInputAttrList);
        resourceList->reconciledInputAttrList = NULL;
    }

    if (resourceList->inputConflictList != NULL) {
        for (ii = 0; ii < resourceList->numInputTensors; ii++) {
            if((resourceList->inputConflictList)[ii] != NULL) {
                NvSciBufAttrListFree((resourceList->inputConflictList)[ii]);
                (resourceList->inputConflictList)[ii] = NULL;
            }
        }
        free(resourceList->inputConflictList);
        resourceList->inputConflictList = NULL;
    }

    if (resourceList->inputAttrList != NULL) {
        for (ii = 0; ii < resourceList->numInputTensors; ii++) {
            if((resourceList->inputAttrList)[ii] != NULL) {
                NvSciBufAttrListFree((resourceList->inputAttrList)[ii]);
                (resourceList->inputAttrList)[ii] = NULL;
            }
        }
        free(resourceList->inputAttrList);
        resourceList->inputAttrList = NULL;
    }

    if (resourceList->reconciledOutputAttrList != NULL) {
        for (ii = 0; ii < resourceList->numOutputTensors; ii++) {
            if((resourceList->reconciledOutputAttrList)[ii] != NULL) {
                NvSciBufAttrListFree((resourceList->reconciledOutputAttrList)[ii]);
                (resourceList->reconciledOutputAttrList)[ii] = NULL;
            }
        }
        free(resourceList->reconciledOutputAttrList);
        resourceList->reconciledOutputAttrList = NULL;
    }

    if (resourceList->outputConflictList != NULL) {
        for (ii = 0; ii < resourceList->numOutputTensors; ii++) {
            if((resourceList->outputConflictList)[ii] != NULL) {
                NvSciBufAttrListFree((resourceList->outputConflictList)[ii]);
                (resourceList->outputConflictList)[ii] = NULL;
            }
        }
        free(resourceList->outputConflictList);
        resourceList->outputConflictList = NULL;
    }

    if (resourceList->outputAttrList != NULL) {
        for (ii = 0; ii < resourceList->numOutputTensors; ii++) {
            if((resourceList->outputAttrList)[ii] != NULL) {
                NvSciBufAttrListFree((resourceList->outputAttrList)[ii]);
                (resourceList->outputAttrList)[ii] = NULL;
            }
        }
        free(resourceList->outputAttrList);
        resourceList->outputAttrList = NULL;
    }

    if (resourceList->reconciledStatisticsOutputAttrList != NULL) {
        for (ii = 0; ii < resourceList->numOutputTaskStatistics; ii++) {
            if((resourceList->reconciledStatisticsOutputAttrList)[ii] != NULL) {
                NvSciBufAttrListFree((resourceList->reconciledStatisticsOutputAttrList)[ii]);
                (resourceList->reconciledStatisticsOutputAttrList)[ii] = NULL;
            }
        }
        free(resourceList->reconciledStatisticsOutputAttrList);
        resourceList->reconciledStatisticsOutputAttrList = NULL;
    }

    if (resourceList->statisticsOutputConflictList != NULL) {
        for (ii = 0; ii < resourceList->numOutputTaskStatistics; ii++) {
            if((resourceList->statisticsOutputConflictList)[ii] != NULL) {
                NvSciBufAttrListFree((resourceList->statisticsOutputConflictList)[ii]);
                (resourceList->statisticsOutputConflictList)[ii] = NULL;
            }
        }
        free(resourceList->statisticsOutputConflictList);
        resourceList->statisticsOutputConflictList = NULL;
    }

    if (resourceList->statisticsOutputAttrList != NULL) {
        for (ii = 0; ii < resourceList->numOutputTaskStatistics; ii++) {
            if((resourceList->statisticsOutputAttrList)[ii] != NULL) {
                NvSciBufAttrListFree((resourceList->statisticsOutputAttrList)[ii]);
                (resourceList->statisticsOutputAttrList)[ii] = NULL;
            }
        }
        free(resourceList->statisticsOutputAttrList);
        resourceList->statisticsOutputAttrList = NULL;
    }

    if (resourceList->outputBufObjRegPtr != NULL) {
        free(resourceList->outputBufObjRegPtr);
        resourceList->outputBufObjRegPtr = NULL;
    }

    if (resourceList->statisticsBufObjRegPtr != NULL) {
        free(resourceList->statisticsBufObjRegPtr);
        resourceList->statisticsBufObjRegPtr = NULL;
    }

    if (resourceList->inputBufObjRegPtr != NULL) {
        free(resourceList->inputBufObjRegPtr);
        resourceList->inputBufObjRegPtr = NULL;
    }

    if (resourceList->bufModule != NULL) {
        NvSciBufModuleClose(resourceList->bufModule);
        resourceList->bufModule = NULL;
    }

    NvSciSyncFenceClear(&(resourceList->eofFence));
    if (resourceList->syncObj != NULL) {
        NvSciSyncObjFree(resourceList->syncObj);
        resourceList->syncObj = NULL;
    }

    if (resourceList->nvSciSyncConflictListObj != NULL) {
        NvSciSyncAttrListFree(resourceList->nvSciSyncConflictListObj);
        resourceList->nvSciSyncConflictListObj = NULL;
    }

    if (resourceList->nvSciSyncReconciledListObj != NULL) {
        NvSciSyncAttrListFree(resourceList->nvSciSyncReconciledListObj);
        resourceList->nvSciSyncReconciledListObj = NULL;
    }

    if (resourceList->signalerAttrListObj != NULL) {
        NvSciSyncAttrListFree(resourceList->signalerAttrListObj);
        resourceList->signalerAttrListObj = NULL;
    }

    if (resourceList->waiterAttrListObj != NULL) {
        NvSciSyncAttrListFree(resourceList->waiterAttrListObj);
        resourceList->waiterAttrListObj = NULL;
    }

    if (resourceList->nvSciCtx != NULL) {
        NvSciSyncCpuWaitContextFree(resourceList->nvSciCtx);
        resourceList->nvSciCtx = NULL;
    }

    if (resourceList->syncModule != NULL) {
        NvSciSyncModuleClose(resourceList->syncModule);
        resourceList->syncModule = NULL;
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

    resourceList->numInputTensors = 0;
    resourceList->numOutputTensors = 0;
    resourceList->numOutputTaskStatistics = 0;
}

cudlaStatus createAndSetAttrList(NvSciBufModule module,
                                 uint64_t bufSize,
                                 NvSciBufAttrList *attrList);


cudlaStatus createAndSetAttrList(NvSciBufModule module,
                                 uint64_t bufSize,
                                 NvSciBufAttrList *attrList) {
    cudlaStatus status = cudlaSuccess;
    NvSciError sciStatus = NvSciError_Success;

    sciStatus = NvSciBufAttrListCreate(module, attrList);
    if (sciStatus != NvSciError_Success) {
        status = cudlaErrorNvSci;
        DPRINTF("Error in creating NvSciBuf attribute list\n");
        return status;
    }

    // TODO: Refactor into multiple dimensions
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
    if (sciStatus != NvSciError_Success)
    {
        status = cudlaErrorNvSci;
        DPRINTF("Error in setting NvSciBuf attribute list\n");
        return status;
    }

    return status;
}

NvSciError fillCpuWaiterAttrList(NvSciSyncAttrList list);

NvSciError fillCpuWaiterAttrList(NvSciSyncAttrList list) {
    bool cpuWaiter = true;
    NvSciSyncAttrKeyValuePair keyValue[2];
    memset(keyValue, 0, sizeof(keyValue));
    keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*) &cpuWaiter;
    keyValue[0].len = sizeof(cpuWaiter);
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*) &cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);
    return NvSciSyncAttrListSetAttrs(list, keyValue, 2);
}

int main(int argc, char** argv) {
    cudlaDevHandle devHandle;
    cudlaModule moduleHandle;
    cudlaStatus err;
    uint32_t statSupport = 0;
    uint32_t dlaFreqInMHz = 0;
    FILE* fp = NULL;
    struct stat st;
    size_t file_size;
    size_t actually_read = 0;
    unsigned char *loadableData = NULL;
    char filename[MAX_FILENAME_LEN];
    const char* suffix = ".csv";


    ResourceList resourceList;

    memset(&resourceList, 0x00, sizeof(ResourceList));

    if ((argc != 4) && (argc != 5)) {
        DPRINTF("Usage : ./test_cudla_layerwise_stats_L0_standalone_test1 <loadable> <freqMHZ> <statSupport> <filenamePrefix>\n");
        return 1;
    }

    if (argc == 5) {
        if((strlen(argv[4])) > (MAX_FILENAME_LEN - RESERVED_SUFFIX_LEN)) {
            DPRINTF("Filename prefix length is too big, greater than maximum permissible prefix length of %u \n",(MAX_FILENAME_LEN - RESERVED_SUFFIX_LEN));
            return 1;
        }
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

    dlaFreqInMHz = atoi(argv[2]);
    statSupport = atoi(argv[3]);

    loadableData = (unsigned char *)malloc(file_size);
    if (loadableData == NULL) {
        DPRINTF("Cannot Allocate memory for loadable\n");
        return 1;
    }

    actually_read = fread(loadableData, 1, file_size, fp);
    if ( actually_read != file_size ) {
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

    err = cudlaModuleLoadFromMemory(devHandle, loadableData, file_size, &moduleHandle, 0);
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
    uint32_t numOutputTaskStatistics = 0;

    cudlaModuleAttribute attribute;

    err = cudlaModuleGetAttributes(moduleHandle, CUDLA_NUM_INPUT_TENSORS, &attribute);
    if (err != cudlaSuccess) {
        DPRINTF("Error in getting numInputTensors = %d\n", err);
        cleanUp(&resourceList);
        return 1;
    }
    numInputTensors = attribute.numInputTensors;
    DPRINTF("numInputTensors = %d\n", numInputTensors);

    err = cudlaModuleGetAttributes(moduleHandle, CUDLA_NUM_OUTPUT_TENSORS, &attribute);
    if (err != cudlaSuccess) {
        DPRINTF("Error in getting numOutputTensors = %d\n", err);
        cleanUp(&resourceList);
        return 1;
    }
    numOutputTensors = attribute.numOutputTensors;
    DPRINTF("numOutputTensors = %d\n", numOutputTensors);

    // using the same attributes to get num_output_task_statistics_tensors
    attribute.numOutputTensors = 0;

    err = cudlaModuleGetAttributes(moduleHandle, CUDLA_NUM_OUTPUT_TASK_STATISTICS, &attribute);
    if (err != cudlaSuccess) {
        DPRINTF("Error in getting numOutputTensors = %d\n", err);
        cleanUp(&resourceList);
        return 1;
    }
    numOutputTaskStatistics = attribute.numOutputTensors;
    DPRINTF("numOutputTaskStatistics = %d\n", numOutputTaskStatistics);

    if(numOutputTaskStatistics == 0) {
        DPRINTF("Layerwise stats is not supported for this Loadable \n");
        cleanUp(&resourceList);
        return 1;
    }

    resourceList.numInputTensors = numInputTensors;
    resourceList.numOutputTensors = numOutputTensors;
    resourceList.numOutputTaskStatistics = numOutputTaskStatistics;

    cudlaModuleTensorDescriptor* inputTensorDesc =
        (cudlaModuleTensorDescriptor*)malloc(sizeof(cudlaModuleTensorDescriptor)*numInputTensors);
    cudlaModuleTensorDescriptor* outputTensorDesc =
        (cudlaModuleTensorDescriptor*)malloc(sizeof(cudlaModuleTensorDescriptor)*numOutputTensors);

    if ((inputTensorDesc == NULL) || (outputTensorDesc == NULL)) {
        if (inputTensorDesc != NULL)
        {
            free(inputTensorDesc);
            inputTensorDesc = NULL;
        }

        if (outputTensorDesc != NULL)
        {
            free(outputTensorDesc);
            outputTensorDesc = NULL;
        }

        cleanUp(&resourceList);
        return 1;
    }

    resourceList.inputTensorDesc = inputTensorDesc;
    resourceList.outputTensorDesc = outputTensorDesc;

    cudlaModuleTensorDescriptor* outputTaskStatisticsDesc =
    (cudlaModuleTensorDescriptor*)malloc(sizeof(cudlaModuleTensorDescriptor)*numOutputTaskStatistics);
    if (outputTaskStatisticsDesc == NULL) {
        free(outputTaskStatisticsDesc);
        outputTaskStatisticsDesc = NULL;
        cleanUp(&resourceList);
        return 1;
    }

    resourceList.outputTaskStatisticsDesc = outputTaskStatisticsDesc;

    attribute.inputTensorDesc = inputTensorDesc;
    err = cudlaModuleGetAttributes(moduleHandle,
                                   CUDLA_INPUT_TENSOR_DESCRIPTORS,
                                   &attribute);
    if (err != cudlaSuccess) {
        DPRINTF("Error in getting input tensor descriptor = %d\n", err);
        cleanUp(&resourceList);
        return 1;
    }
    DPRINTF("Printing input tensor descriptor\n");
    printTensorDesc(inputTensorDesc);

    attribute.outputTensorDesc = outputTensorDesc;
    err = cudlaModuleGetAttributes(moduleHandle,
                                   CUDLA_OUTPUT_TENSOR_DESCRIPTORS,
                                   &attribute);
    if (err != cudlaSuccess) {
        DPRINTF("Error in getting output tensor descriptor = %d\n", err);
        cleanUp(&resourceList);
        return 1;
    }
    DPRINTF("Printing output tensor descriptor\n");
    printTensorDesc(outputTensorDesc);

    attribute.outputTensorDesc = outputTaskStatisticsDesc;
    err = cudlaModuleGetAttributes(moduleHandle,
                                   CUDLA_OUTPUT_TASK_STATISTICS_DESCRIPTORS,
                                   &attribute);
    if (err != cudlaSuccess) {
        DPRINTF("Error in getting task statistics descriptor = %d\n", err);
        cleanUp(&resourceList);
        return 1;
    }
    DPRINTF("Printing output task statistics descriptor size\n");
    for (uint32_t ii = 0; ii < numOutputTaskStatistics; ii++) {
        DPRINTF("The size of %u descriptor is %lu\n", ii,outputTaskStatisticsDesc[ii].size);
    }

    // Setup the input and output buffers.
    unsigned char** inputBuffer = (unsigned char **)malloc(sizeof(unsigned char *)*numInputTensors);
    if (inputBuffer == NULL) {
        DPRINTF("Error in allocating memory for input buffer array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(inputBuffer, 0x00, sizeof(unsigned char *)*numInputTensors);
    resourceList.inputBuffer = inputBuffer;

    for (uint32_t ii = 0; ii < numInputTensors; ii++) {
        inputBuffer[ii] = (unsigned char* )malloc(inputTensorDesc[ii].size);
        if (inputBuffer[ii] == NULL) {
            DPRINTF("Error in allocating input memory\n");
            cleanUp(&resourceList);
            return 1;
        }
        memset(inputBuffer[ii], 0x01, inputTensorDesc[ii].size);
    }

    unsigned char** outputBuffer = (unsigned char **)malloc(sizeof(unsigned char *)*numOutputTensors);
    if (outputBuffer == NULL) {
        DPRINTF("Error in allocating memory for output buffer array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(outputBuffer, 0x00, sizeof(unsigned char *)*numOutputTensors);
    resourceList.outputBuffer = outputBuffer;

    for (uint32_t ii = 0; ii < numOutputTensors; ii++) {
        outputBuffer[ii] = (unsigned char* )malloc(outputTensorDesc[ii].size);
        if (outputBuffer[ii] == NULL) {
            DPRINTF("Error in allocating output memory\n");
            cleanUp(&resourceList);
            return 1;
        }
        memset(outputBuffer[ii], 0x00, outputTensorDesc[ii].size);
    }

    unsigned char** statisticsOutputBuffer = (unsigned char **)malloc(sizeof(unsigned char *)*numOutputTaskStatistics);
    if (statisticsOutputBuffer == NULL) {
        DPRINTF("Error in allocating memory for output buffer array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(statisticsOutputBuffer, 0x00, sizeof(unsigned char *)*numOutputTaskStatistics);
    resourceList.statisticsOutputBuffer = statisticsOutputBuffer;

    for (uint32_t ii = 0; ii < numOutputTaskStatistics; ii++) {
        statisticsOutputBuffer[ii] = (unsigned char* )malloc(outputTaskStatisticsDesc[ii].size);
        if (outputBuffer[ii] == NULL) {
            DPRINTF("Error in allocating output memory\n");
            cleanUp(&resourceList);
            return 1;
        }
        memset(statisticsOutputBuffer[ii], 0x00, outputTaskStatisticsDesc[ii].size);
    }

    NvSciBufModule bufModule = NULL;
    NvSciBufAttrList *inputAttrList = {NULL};
    NvSciBufAttrList *outputAttrList = {NULL};
    NvSciBufAttrList *statisticsOutputAttrList = {NULL};
    NvSciBufAttrList *reconciledInputAttrList = {NULL};
    NvSciBufAttrList *reconciledOutputAttrList = {NULL};
    NvSciBufAttrList *reconciledStatisticsOutputAttrList = {NULL};
    NvSciBufAttrList *inputConflictList = {NULL};
    NvSciBufAttrList *outputConflictList = {NULL};
    NvSciBufAttrList *statisticsOutputConflictList = {NULL};
    NvSciError sciError = NvSciError_Success;

    sciError = NvSciBufModuleOpen(&bufModule);
    if (sciError != NvSciError_Success) {
        DPRINTF("Error in initializing NvSciBufModule\n");
        cleanUp(&resourceList);
        return 1;
    }
    resourceList.bufModule = bufModule;

    // creating and setting input attribute list

    inputAttrList = (NvSciBufAttrList *)malloc(sizeof(NvSciBufAttrList)*numInputTensors);
    if (inputAttrList == NULL) {
        DPRINTF("Error in allocating memory for input buffer array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(inputAttrList, 0x00, sizeof(NvSciBufAttrList)*numInputTensors);
    resourceList.inputAttrList = inputAttrList;

    reconciledInputAttrList = (NvSciBufAttrList *)malloc(sizeof(NvSciBufAttrList)*numInputTensors);
    if (reconciledInputAttrList == NULL) {
        DPRINTF("Error in allocating memory for input buffer array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(reconciledInputAttrList, 0x00, sizeof(NvSciBufAttrList)*numInputTensors);
    resourceList.reconciledInputAttrList = reconciledInputAttrList;

    inputConflictList = (NvSciBufAttrList *)malloc(sizeof(NvSciBufAttrList)*numInputTensors);
    if (inputConflictList == NULL) {
        DPRINTF("Error in allocating memory for input buffer array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(inputConflictList, 0x00, sizeof(NvSciBufAttrList)*numInputTensors);
    resourceList.inputConflictList = inputConflictList;


    for (uint32_t ii = 0; ii < numInputTensors; ii++) {
        err = createAndSetAttrList(bufModule,
                                   inputTensorDesc[ii].size,
                                   &inputAttrList[ii]);
        if (err != cudlaSuccess) {
            DPRINTF("Error in creating NvSciBuf attribute list for input attribute\n");
            cleanUp(&resourceList);
            return 1;
        }

        sciError = NvSciBufAttrListReconcile(&inputAttrList[ii],
                                             1,
                                             &reconciledInputAttrList[ii],
                                             &inputConflictList[ii]);
        if (sciError != NvSciError_Success) {
            DPRINTF("Error in reconciling NvSciBuf attribute list for input attribute\n");
            cleanUp(&resourceList);
            return 1;
        }

    }

    outputAttrList = (NvSciBufAttrList *)malloc(sizeof(NvSciBufAttrList)*numOutputTensors);
    if (outputAttrList == NULL) {
        DPRINTF("Error in allocating memory for input buffer array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(outputAttrList, 0x00, sizeof(NvSciBufAttrList)*numOutputTensors);
    resourceList.outputAttrList = outputAttrList;

    reconciledOutputAttrList = (NvSciBufAttrList *)malloc(sizeof(NvSciBufAttrList)*numOutputTensors);
    if (reconciledOutputAttrList == NULL) {
        DPRINTF("Error in allocating memory for input buffer array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(reconciledOutputAttrList, 0x00, sizeof(NvSciBufAttrList)*numOutputTensors);
    resourceList.reconciledOutputAttrList = reconciledOutputAttrList;

    outputConflictList = (NvSciBufAttrList *)malloc(sizeof(NvSciBufAttrList)*numOutputTensors);
    if (outputConflictList == NULL) {
        DPRINTF("Error in allocating memory for input buffer array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(outputConflictList, 0x00, sizeof(NvSciBufAttrList)*numOutputTensors);
    resourceList.outputConflictList = outputConflictList;

    // creating and setting output attribute list
    for (uint32_t ii = 0; ii < numOutputTensors; ii++) {
        err = createAndSetAttrList(bufModule,
                                   outputTensorDesc[ii].size,
                                   &outputAttrList[ii]);
        if (err != cudlaSuccess) {
            DPRINTF("Error in creating NvSciBuf attribute list for output attibute\n");
            cleanUp(&resourceList);
            return 1;
        }

        sciError = NvSciBufAttrListReconcile(&outputAttrList[ii],
                                             1,
                                             &reconciledOutputAttrList[ii],
                                             &outputConflictList[ii]);
        if (sciError != NvSciError_Success) {
            DPRINTF("Error in reconciling NvSciBuf attribute list for output attribute\n");
            cleanUp(&resourceList);
            return 1;
        }
    }

    statisticsOutputAttrList = (NvSciBufAttrList *)malloc(sizeof(NvSciBufAttrList)*numOutputTaskStatistics);
    if (statisticsOutputAttrList == NULL) {
        DPRINTF("Error in allocating memory for input buffer array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(statisticsOutputAttrList, 0x00, sizeof(NvSciBufAttrList)*numOutputTaskStatistics);
    resourceList.statisticsOutputAttrList = statisticsOutputAttrList;

    reconciledStatisticsOutputAttrList = (NvSciBufAttrList *)malloc(sizeof(NvSciBufAttrList)*numOutputTaskStatistics);
    if (reconciledStatisticsOutputAttrList == NULL) {
        DPRINTF("Error in allocating memory for input buffer array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(reconciledStatisticsOutputAttrList, 0x00, sizeof(NvSciBufAttrList)*numOutputTaskStatistics);
    resourceList.reconciledStatisticsOutputAttrList = reconciledStatisticsOutputAttrList;

    statisticsOutputConflictList = (NvSciBufAttrList *)malloc(sizeof(NvSciBufAttrList)*numOutputTaskStatistics);
    if (statisticsOutputConflictList == NULL) {
        DPRINTF("Error in allocating memory for input buffer array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(statisticsOutputConflictList, 0x00, sizeof(NvSciBufAttrList)*numOutputTaskStatistics);
    resourceList.statisticsOutputConflictList = statisticsOutputConflictList;

    // creating and setting statistics output attribute list
    for (uint32_t ii = 0; ii < numOutputTaskStatistics; ii++) {
        err = createAndSetAttrList(bufModule,
                                   outputTaskStatisticsDesc[ii].size,
                                   &statisticsOutputAttrList[ii]);
        if (err != cudlaSuccess) {
            DPRINTF("Error in creating NvSciBuf attribute list\n");
            cleanUp(&resourceList);
            return 1;
        }

        sciError = NvSciBufAttrListReconcile(&statisticsOutputAttrList[ii],
                                             1,
                                             &reconciledStatisticsOutputAttrList[ii],
                                             &statisticsOutputConflictList[ii]);
        if (sciError != NvSciError_Success) {
            DPRINTF("Error in reconciling NvSciBuf attribute list\n");
            cleanUp(&resourceList);
            return 1;
        }
    }

    NvSciBufObj *inputBufObj = (NvSciBufObj *)malloc(sizeof(NvSciBufObj)*numInputTensors);
    NvSciBufObj *outputBufObj = (NvSciBufObj *)malloc(sizeof(NvSciBufObj)*numOutputTensors);
    NvSciBufObj *statisticsBufObj = (NvSciBufObj *)malloc(sizeof(NvSciBufObj)*numOutputTaskStatistics);

    resourceList.inputBufObj = inputBufObj;
    resourceList.outputBufObj = outputBufObj;
    resourceList.statisticsBufObj = statisticsBufObj;

    for (uint32_t ii = 0; ii < numInputTensors; ii++) {
        sciError = NvSciBufObjAlloc(reconciledInputAttrList[ii], &inputBufObj[ii]);
        if (sciError != NvSciError_Success) {
            DPRINTF("Error in allocating NvSciBuf object\n");
            cleanUp(&resourceList);
            return 1;
        }
    }

    for (uint32_t ii = 0; ii < numOutputTensors; ii++) {
        sciError = NvSciBufObjAlloc(reconciledOutputAttrList[ii], &outputBufObj[ii]);
        if (sciError != NvSciError_Success) {
            DPRINTF("Error in allocating NvSciBuf object\n");
            cleanUp(&resourceList);
            return 1;
        }
    }

    for (uint32_t ii = 0; ii < numOutputTaskStatistics; ii++) {
        sciError = NvSciBufObjAlloc(reconciledStatisticsOutputAttrList[ii], &statisticsBufObj[ii]);
        if (sciError != NvSciError_Success) {
            DPRINTF("Error in allocating NvSciBuf object\n");
            cleanUp(&resourceList);
            return 1;
        }
    }

    uint64_t** inputBufObjRegPtr = (uint64_t **)malloc(sizeof(uint64_t*)*numInputTensors);
    uint64_t** outputBufObjRegPtr = (uint64_t **)malloc(sizeof(uint64_t*)*numOutputTensors);
    uint64_t** statisticsBufObjRegPtr = (uint64_t **)malloc(sizeof(uint64_t*)*numOutputTaskStatistics);

    if ((inputBufObjRegPtr == NULL) || (outputBufObjRegPtr == NULL) || (statisticsBufObjRegPtr == NULL)) {
        if (inputBufObjRegPtr != NULL) {
            free(inputBufObjRegPtr);
            inputBufObjRegPtr = NULL;
        }

        if (outputBufObjRegPtr != NULL) {
            free(outputBufObjRegPtr);
            outputBufObjRegPtr = NULL;
        }

        if (statisticsBufObjRegPtr != NULL) {
            free(statisticsBufObjRegPtr);
            statisticsBufObjRegPtr = NULL;
        }

        cleanUp(&resourceList);
        return 1;
    }

    resourceList.inputBufObjRegPtr = inputBufObjRegPtr;
    resourceList.outputBufObjRegPtr = outputBufObjRegPtr;
    resourceList.statisticsBufObjRegPtr = statisticsBufObjRegPtr;

    void **inputBufObjBuffer = (void **)malloc(sizeof(void*)*numInputTensors);
    void **outputBufObjBuffer = (void **)malloc(sizeof(void*)*numOutputTensors);
    void **statisticsBufObjBuffer = (void **)malloc(sizeof(void*)*numOutputTaskStatistics);

    cudlaExternalMemoryHandleDesc memDesc = { 0 };
    // importing external memory
    for (uint32_t ii = 0; ii < numInputTensors; ii++) {
        memset(&memDesc, 0, sizeof(memDesc));
        memDesc.extBufObject = (void *)inputBufObj[ii];
        memDesc.size = inputTensorDesc[ii].size;
        err = cudlaImportExternalMemory(devHandle, &memDesc, &inputBufObjRegPtr[ii], 0);
        if (err != cudlaSuccess) {
            DPRINTF("Error in importing external memory = %d\n", err);
            cleanUp(&resourceList);
            return 1;
        }

        sciError = NvSciBufObjGetCpuPtr(inputBufObj[ii], &inputBufObjBuffer[ii]);
        if (sciError != NvSciError_Success) {
            DPRINTF("Error in getting NvSciBuf CPU pointer\n");
            cleanUp(&resourceList);
            return 1;
        }
        memcpy(inputBufObjBuffer[ii], inputBuffer[ii], inputTensorDesc[ii].size);
    }

    for (uint32_t ii = 0; ii < numOutputTensors; ii++) {
        memset(&memDesc, 0, sizeof(memDesc));
        memDesc.extBufObject = (void *)outputBufObj[ii];
        memDesc.size = outputTensorDesc[ii].size;
        err = cudlaImportExternalMemory(devHandle, &memDesc, &outputBufObjRegPtr[ii], 0);
        if (err != cudlaSuccess) {
            DPRINTF("Error in importing external memory = %d\n", err);
            cleanUp(&resourceList);
            return 1;
        }

        sciError = NvSciBufObjGetCpuPtr(outputBufObj[ii], &outputBufObjBuffer[ii]);
        if (sciError != NvSciError_Success) {
            DPRINTF("Error in getting NvSciBuf CPU pointer\n");
            cleanUp(&resourceList);
            return 1;
        }
        memset(outputBufObjBuffer[ii], 0, outputTensorDesc[ii].size);
    }

    for (uint32_t ii = 0; ii < numOutputTaskStatistics; ii++) {
        memset(&memDesc, 0, sizeof(memDesc));
        memDesc.extBufObject = (void *)statisticsBufObj[ii];
        memDesc.size = outputTaskStatisticsDesc[ii].size;
        err = cudlaImportExternalMemory(devHandle, &memDesc, &statisticsBufObjRegPtr[ii], CUDLA_TASK_STATISTICS);
        if (err != cudlaSuccess) {
            DPRINTF("Error in importing external memory = %d\n", err);
            cleanUp(&resourceList);
            return 1;
        }

        sciError = NvSciBufObjGetCpuPtr(statisticsBufObj[ii], &statisticsBufObjBuffer[ii]);
        if (sciError != NvSciError_Success) {
            DPRINTF("Error in getting NvSciBuf CPU pointer\n");
            cleanUp(&resourceList);
            return 1;
        }
        memset(statisticsBufObjBuffer[ii], 0, outputTaskStatisticsDesc[ii].size);
    }

    uint64_t *outputStatisticsBufferRegisteredPtr[numOutputTensors + numOutputTaskStatistics] = {0} ;

    uint32_t index = 0;
    for (; index < numOutputTensors ; index++) {
        outputStatisticsBufferRegisteredPtr[index] = ((outputBufObjRegPtr[index]));
    }

    for (uint32_t jj=0; jj < numOutputTaskStatistics ; jj++) {
        outputStatisticsBufferRegisteredPtr[index++] = ((statisticsBufObjRegPtr[jj]));
    }

    NvSciSyncObj syncObj;
    NvSciSyncModule syncModule;
    NvSciSyncAttrList syncAttrListObj[2];
    NvSciSyncCpuWaitContext nvSciCtx;
    NvSciSyncAttrList waiterAttrListObj = NULL;
    NvSciSyncAttrList signalerAttrListObj = NULL;
    NvSciSyncAttrList nvSciSyncConflictListObj;
    NvSciSyncAttrList nvSciSyncReconciledListObj;
    
    sciError = NvSciSyncModuleOpen(&syncModule);
    if (sciError != NvSciError_Success) {
        DPRINTF("Error in initializing NvSciSyncModuleOpen\n");
        cleanUp(&resourceList);
        return 1;
    }
    resourceList.syncModule = syncModule;
    
    sciError = NvSciSyncCpuWaitContextAlloc(syncModule, &nvSciCtx);
    if (sciError != NvSciError_Success) {
        DPRINTF("Error in allocating cpu wait context NvSciSyncCpuWaitContextAlloc\n");
        cleanUp(&resourceList);
        return 1;
    }
    resourceList.nvSciCtx = nvSciCtx;
    
    sciError = NvSciSyncAttrListCreate(syncModule, &signalerAttrListObj);
    if (sciError != NvSciError_Success) {
        DPRINTF("Error in creating NvSciSync attribute list\n");
        cleanUp(&resourceList);
        return 1;
    }
    resourceList.signalerAttrListObj = signalerAttrListObj;
    
    sciError = NvSciSyncAttrListCreate(syncModule, &waiterAttrListObj);
    if (sciError != NvSciError_Success) {
        DPRINTF("Error in creating NvSciSync attribute list\n");
        cleanUp(&resourceList);
        return 1;
    }
    resourceList.waiterAttrListObj = waiterAttrListObj;
    
    err = cudlaGetNvSciSyncAttributes(reinterpret_cast<uint64_t*>(signalerAttrListObj),
                                      CUDLA_NVSCISYNC_ATTR_SIGNAL);
    if (err != cudlaSuccess) {
        DPRINTF("Error in getting cuDLA's NvSciSync attributes\n");
        cleanUp(&resourceList);
        return 1;
    }
    
    sciError = fillCpuWaiterAttrList(waiterAttrListObj);
    if (sciError != NvSciError_Success) {
        DPRINTF("Error in setting NvSciSync attribute list\n");
        cleanUp(&resourceList);
        return 1;
    }
    
    syncAttrListObj[0] = signalerAttrListObj;
    syncAttrListObj[1] = waiterAttrListObj;
    sciError = NvSciSyncAttrListReconcile(syncAttrListObj,
                                          2,
                                          &nvSciSyncReconciledListObj,
                                          &nvSciSyncConflictListObj);
    if (sciError != NvSciError_Success) {
        DPRINTF("Error in reconciling NvSciSync's attribute lists\n");
        cleanUp(&resourceList);
        return 1;
    }
    resourceList.nvSciSyncConflictListObj = nvSciSyncConflictListObj;
    resourceList.nvSciSyncReconciledListObj = nvSciSyncReconciledListObj;
    
    sciError = NvSciSyncObjAlloc(nvSciSyncReconciledListObj, &syncObj); 
    if (sciError != NvSciError_Success) {
        DPRINTF("Error in allocating NvSciSync object\n");
        cleanUp(&resourceList);
        return 1;
    }
    resourceList.syncObj = syncObj;
    
    // importing external semaphore
    uint64_t* nvSciSyncObjRegPtr = NULL;
    cudlaExternalSemaphoreHandleDesc semaMemDesc = { 0 };
    memset(&semaMemDesc, 0, sizeof(semaMemDesc));
    semaMemDesc.extSyncObject = syncObj;
    err = cudlaImportExternalSemaphore(devHandle,
                                       &semaMemDesc,
                                       &nvSciSyncObjRegPtr,
                                       0);
    if (err != cudlaSuccess) {
        DPRINTF("Error in importing external semaphore = %d\n", err);
        cleanUp(&resourceList);
        return 1;
    }
    DPRINTF("ALL MEMORY REGISTERED SUCCESSFULLY\n");
    
    // Signal Events
    cudlaSignalEvents* signalEvents;
    signalEvents = (cudlaSignalEvents *)malloc(sizeof(cudlaSignalEvents));
    if (signalEvents == NULL) {
        DPRINTF("Error in allocating signal events\n");
        cleanUp(&resourceList);
        return 1;
    }

    signalEvents->numEvents = 1;
    uint64_t** devPtrs = (uint64_t **)malloc(signalEvents->numEvents *
                                             sizeof(uint64_t *));
    if (devPtrs == NULL) {
        DPRINTF("Error in allocating output pointer's array of registered objects\n");
        cleanUp(&resourceList);
        return 1;
    }
    devPtrs[0] = nvSciSyncObjRegPtr;
    signalEvents->devPtrs = devPtrs;
    resourceList.devPtrs = devPtrs;
    
    signalEvents->eofFences = (CudlaFence *)malloc(signalEvents->numEvents *
                                                   sizeof(CudlaFence));
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
    task.outputTensor = (uint64_t * const*)&outputStatisticsBufferRegisteredPtr;

    if(statSupport == 1) {
        task.numOutputTensors = (numOutputTensors + numOutputTaskStatistics);
        DPRINTF("Layerwise profiling is requested \n");
    } else {
        task.numOutputTensors = numOutputTensors;
        DPRINTF("Layerwise profiling is not requested \n");
    }

    task.numInputTensors = numInputTensors;
    task.inputTensor = inputBufObjRegPtr;
    task.waitEvents = NULL;
    task.signalEvents = signalEvents;

    err = cudlaSubmitTask(devHandle, &task, 1, NULL, 0);
    if (err != cudlaSuccess) {
        DPRINTF("Error in submitting task\n");
        cleanUp(&resourceList);
        return 1;
    }
    DPRINTF("SUBMIT IS DONE !!!\n");

    // Wait for operations to finish and bring output buffer to CPU.
    sciError = NvSciSyncFenceWait(reinterpret_cast<NvSciSyncFence*>(signalEvents->eofFences[0].fence),
                                  nvSciCtx, -1);
    if (sciError != NvSciError_Success) {
        DPRINTF("Error in waiting on NvSciSyncFence\n");
        cleanUp(&resourceList);
        return 1;
    }

    // copy statistics data to cpu
    for (uint32_t ii = 0; ii < numOutputTensors; ii++) {
        memcpy(outputBuffer[ii], outputBufObjBuffer[ii], outputTensorDesc[ii].size);
    }

    if(statSupport == 1) {
        for (uint32_t ii = 0; ii < numOutputTaskStatistics; ii++) {
            memcpy(statisticsOutputBuffer[ii], statisticsBufObjBuffer[ii], outputTaskStatisticsDesc[ii].size);
        }

        const cudlaExternalEtbl* etbl = NULL;
        if (cudlaGetExternalExportTable(&etbl,0) != cudlaSuccess) {
            DPRINTF("Error in getting export table\n");
            cleanUp(&resourceList);
            return 1;
        }

        void** csv = (void **)malloc(sizeof(void *)*numOutputTaskStatistics);
        if (csv == NULL) {
            DPRINTF("Error in allocating memory for csv stream\n");
            cleanUp(&resourceList);
            return 1;
        }
        memset(csv, 0x00, sizeof(void *)*numOutputTaskStatistics);
        resourceList.csv = csv;
        for (uint32_t ii = 0; ii < numOutputTaskStatistics; ii++) {
            cudlaTranslateCsvAttribute csvAttribute;
            uint64_t csvStreamLength = 0;

            err = etbl->etiTranslateStats(devHandle,statisticsOutputBuffer[ii],dlaFreqInMHz,ii,CUDLA_GET_CSV_LENGTH,&csvAttribute);
            csv[ii] = (void* )malloc(csvAttribute.csvStreamLength);
            csvStreamLength = csvAttribute.csvStreamLength;
            DPRINTF("size for statistics buffer %u is %lu \n",ii,csvStreamLength);

            if (csv[ii] == NULL) {
                DPRINTF("Error in allocating memory for csv stream\n");
                cleanUp(&resourceList);
                return 1;
            }
            memset(csv[ii], 0x00, csvAttribute.csvStreamLength);

            csvAttribute.csvStreamStats = csv[ii];
            err = etbl->etiTranslateStats(devHandle,statisticsOutputBuffer[ii],dlaFreqInMHz,ii,CUDLA_GET_CSV_STATS,&csvAttribute);
            if (err != cudlaSuccess) {
                DPRINTF("Error in translating stats\n");
                cleanUp(&resourceList);
                return 1;
            }

            if (argc == 5) {
                sprintf(filename,"%s%u%s", argv[4],(ii+1),suffix);
                fp = fopen(filename, "w+");
                if (fp == NULL) {
                    DPRINTF("Cannot open file %s\n", filename);
                    cleanUp(&resourceList);
                    return 1;
                }

                uint32_t ret_val = fwrite(csv[ii],sizeof(char),csvStreamLength,fp);
                if(ret_val != csvStreamLength) {
                    DPRINTF("number of elements written to file is %u \n", ret_val);
                    cleanUp(&resourceList);
                    return 1;
                }

                fclose(fp);
            } else {
                DPRINTF("%s \n",(char *)csv[ii]);
            }
        }
    }

    // unregister the CUDA-allocated buffers.
    for (uint32_t ii = 0; ii < numInputTensors; ii++) {
        err = cudlaMemUnregister(devHandle,
                                 (inputBufObjRegPtr[ii]));
        if (err != cudlaSuccess) {
            DPRINTF("Error in registering input memory = %d\n", err);
            cleanUp(&resourceList);
            return 1;
        }
    }

    for (uint32_t ii = 0; ii < numOutputTensors; ii++) {
        err = cudlaMemUnregister(devHandle,
                                 (outputBufObjRegPtr[ii]));
        if (err != cudlaSuccess) {
            DPRINTF("Error in registering output memory = %d\n", err);
            cleanUp(&resourceList);
            return 1;
        }
    }

    for (uint32_t ii = 0; ii < numOutputTaskStatistics; ii++) {
        err = cudlaMemUnregister(devHandle,
                                 (statisticsBufObjRegPtr[ii]));
        if (err != cudlaSuccess) {
            DPRINTF("Error in registering output memory = %d\n", err);
            cleanUp(&resourceList);
            return 1;
        }
    }

    err = cudlaMemUnregister(devHandle, nvSciSyncObjRegPtr);
    if (err != cudlaSuccess) {
        DPRINTF("Error in unregistering external semaphore = %d\n", err);
        cleanUp(&resourceList);
        return 1;
    }

    DPRINTF("ALL MEMORY UNREGISTERED SUCCESSFULLY\n");


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

    cleanUp(&resourceList);

    DPRINTF("cuDLALayerwiseStatsStandalone DONE !!!\n");

    return 0;
}
