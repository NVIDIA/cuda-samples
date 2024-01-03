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
#include "cuda_runtime.h"
#include "cudlaExternalEtbl.hpp"

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

typedef struct {
    cudlaDevHandle devHandle;
    cudlaModule moduleHandle;
    unsigned char* loadableData;
    cudaStream_t stream;
    uint32_t numInputTensors;
    uint32_t numOutputTensors;
    uint32_t numOutputTaskStatistics;
    unsigned char** inputBuffer;
    unsigned char** outputBuffer;
    unsigned char** statisticsOutputBuffer;
    void** inputBufferGPU;
    void** outputBufferGPU;
    void** outputTaskStatisticsGPU;
    void **csv;
    cudlaModuleTensorDescriptor* inputTensorDesc;
    cudlaModuleTensorDescriptor* outputTensorDesc;
    cudlaModuleTensorDescriptor* outputTaskStatisticsDesc;
    uint64_t** inputBufferRegisteredPtr;
    uint64_t** outputBufferRegisteredPtr;
    uint64_t** outputTaskStatisticsRegisteredPtr;
    uint64_t** outputStatisticsBufferRegisteredPtr;
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

    if (resourceList->inputBufferGPU != NULL) {
        for (ii = 0; ii < resourceList->numInputTensors; ii++) {
            if ((resourceList->inputBufferGPU)[ii] != NULL) {
                cudaFree((resourceList->inputBufferGPU)[ii]);
                (resourceList->inputBufferGPU)[ii] = NULL;
            }
        }
        free(resourceList->inputBufferGPU);
        resourceList->inputBufferGPU = NULL;
    }

    if (resourceList->outputBufferGPU != NULL) {
        for (ii = 0; ii < resourceList->numOutputTensors; ii++) {
            if ((resourceList->outputBufferGPU)[ii] != NULL) {
                cudaFree((resourceList->outputBufferGPU)[ii]);
                (resourceList->outputBufferGPU)[ii] = NULL;
            }
        }
        free(resourceList->outputBufferGPU);
        resourceList->outputBufferGPU = NULL;
    }

    if (resourceList->outputTaskStatisticsGPU != NULL) {
        for (ii = 0; ii < resourceList->numOutputTaskStatistics; ii++) {
            if ((resourceList->outputTaskStatisticsGPU)[ii] != NULL) {
                cudaFree((resourceList->outputTaskStatisticsGPU)[ii]);
                (resourceList->outputTaskStatisticsGPU)[ii] = NULL;
            }
        }
        free(resourceList->outputTaskStatisticsGPU);
        resourceList->outputTaskStatisticsGPU = NULL;
    }

    if (resourceList->csv != NULL) {
        for (ii = 0; ii < resourceList->numOutputTaskStatistics; ii++) {
            if ((resourceList->csv)[ii] != NULL)
            {
                free((resourceList->csv)[ii]);
                (resourceList->csv)[ii] = NULL;
            }
        }
        free(resourceList->csv);
        resourceList->csv = NULL;
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
            if ((resourceList->outputBuffer)[ii] != NULL)
            {
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

    if (resourceList->stream != NULL) {
        cudaStreamDestroy(resourceList->stream);
        resourceList->stream = NULL;
    }

    if (resourceList->inputBufferRegisteredPtr != NULL) {
        free(resourceList->inputBufferRegisteredPtr);
        resourceList->inputBufferRegisteredPtr = NULL;
    }

    if (resourceList->outputBufferRegisteredPtr != NULL) {
        free(resourceList->outputBufferRegisteredPtr);
        resourceList->outputBufferRegisteredPtr = NULL;
    }

    if (resourceList->outputTaskStatisticsRegisteredPtr != NULL) {
        free(resourceList->outputTaskStatisticsRegisteredPtr);
        resourceList->outputTaskStatisticsRegisteredPtr = NULL;
    }

    if (resourceList->outputStatisticsBufferRegisteredPtr != NULL) {
        free(resourceList->outputStatisticsBufferRegisteredPtr);
        resourceList->outputStatisticsBufferRegisteredPtr = NULL;
    }

    resourceList->numInputTensors = 0;
    resourceList->numOutputTensors = 0;
    resourceList->numOutputTaskStatistics = 0;
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

    cudaStream_t stream;
    cudaError_t result;
    const char* errPtr = NULL;

    ResourceList resourceList;

    memset(&resourceList, 0x00, sizeof(ResourceList));

    if ((argc != 4) && (argc != 5)) {
        DPRINTF("Usage : ./test_cudla_layerwise_stats_L0_hybrid_test1 <loadable> <freqMHZ> <statSupport> <filename prefix>\n");
        return 1;
    }

    if (argc == 5) {
        if((strlen(argv[4])) > (MAX_FILENAME_LEN - RESERVED_SUFFIX_LEN))
        {
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

    // Initialize CUDA.
    result = cudaFree(0);
    if (result != cudaSuccess) {
        errPtr = cudaGetErrorName(result);
        DPRINTF("Error in creating cudaFree = %s\n", errPtr);
        cleanUp(&resourceList);
        return 1;
    }

    result = cudaSetDevice(0);
    if (result != cudaSuccess) {
        errPtr = cudaGetErrorName(result);
        DPRINTF("Error in creating cudaSetDevice = %s\n", errPtr);
        cleanUp(&resourceList);
        return 1;
    }

    err = cudlaCreateDevice(0, &devHandle, CUDLA_CUDA_DLA);
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

    // Create CUDA stream.
    result = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    if (result != cudaSuccess) {
        errPtr = cudaGetErrorName(result);
        DPRINTF("Error in creating cuda stream = %s\n", errPtr);
        cleanUp(&resourceList);
        return 1;
    }

    resourceList.stream = stream;

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

    // Setup the input and output buffers which will be used as an input to CUDA.
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

    // Allocate memory on GPU.
    void** inputBufferGPU = (void **)malloc(sizeof(void *)*numInputTensors);
    if (inputBufferGPU == NULL) {
        DPRINTF("Error in allocating memory for input buffer GPU array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(inputBufferGPU, 0x00, sizeof(void *)*numInputTensors);
    resourceList.inputBufferGPU = inputBufferGPU;

    for (uint32_t ii = 0; ii < numInputTensors; ii++) {
        result = cudaMalloc(&(inputBufferGPU[ii]), inputTensorDesc[ii].size);
        if (result != cudaSuccess)
        {
            DPRINTF("Error in allocating input memory on GPU\n");
            cleanUp(&resourceList);
            return 1;
        }
    }

    void** outputBufferGPU = (void **)malloc(sizeof(void *)*numOutputTensors);
    if (outputBufferGPU == NULL) {
        DPRINTF("Error in allocating memory for output buffer GPU array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(outputBufferGPU, 0x00, sizeof(void *)*numOutputTensors);
    resourceList.outputBufferGPU = outputBufferGPU;

    for (uint32_t ii = 0; ii < numOutputTensors; ii++) {
        result = cudaMalloc(&(outputBufferGPU[ii]), outputTensorDesc[ii].size);
        if (result != cudaSuccess) {
            DPRINTF("Error in allocating output memory on GPU\n");
            cleanUp(&resourceList);
            return 1;
        }
    }

    void** outputTaskStatisticsGPU = (void **)malloc(sizeof(void *)*numOutputTaskStatistics);
    if (outputTaskStatisticsGPU == NULL) {
        DPRINTF("Error in allocating memory for output task statistics GPU array\n");
        cleanUp(&resourceList);
        return 1;
    }
    memset(outputTaskStatisticsGPU, 0x00, sizeof(void *)*numOutputTaskStatistics);
    resourceList.outputTaskStatisticsGPU = outputTaskStatisticsGPU;

    for (uint32_t ii = 0; ii < numOutputTaskStatistics; ii++) {
        result = cudaMalloc(&(outputTaskStatisticsGPU[ii]), outputTaskStatisticsDesc[ii].size);
        if (result != cudaSuccess) {
            DPRINTF("Error in allocating task statistics memory on GPU\n");
            cleanUp(&resourceList);
            return 1;
        }
    }

    uint64_t** inputBufferRegisteredPtr = (uint64_t **)malloc(sizeof(uint64_t*)*numInputTensors);
    uint64_t** outputBufferRegisteredPtr = (uint64_t **)malloc(sizeof(uint64_t*)*numOutputTensors);
    uint64_t** outputTaskStatisticsRegisteredPtr = (uint64_t **)malloc(sizeof(uint64_t*)*numOutputTaskStatistics);

    if ((inputBufferRegisteredPtr == NULL) || (outputBufferRegisteredPtr == NULL) || (outputTaskStatisticsRegisteredPtr == NULL)) {
        if (inputBufferRegisteredPtr != NULL) {
            free(inputBufferRegisteredPtr);
            inputBufferRegisteredPtr = NULL;
        }

        if (outputBufferRegisteredPtr != NULL) {
            free(outputBufferRegisteredPtr);
            outputBufferRegisteredPtr = NULL;
        }

        if (outputTaskStatisticsRegisteredPtr != NULL) {
            free(outputTaskStatisticsRegisteredPtr);
            outputTaskStatisticsRegisteredPtr = NULL;
        }

        cleanUp(&resourceList);
        return 1;
    }

    resourceList.inputBufferRegisteredPtr = inputBufferRegisteredPtr;
    resourceList.outputBufferRegisteredPtr = outputBufferRegisteredPtr;
    resourceList.outputTaskStatisticsRegisteredPtr = outputTaskStatisticsRegisteredPtr;

    // Register the CUDA-allocated buffers.
    for (uint32_t ii = 0; ii < numInputTensors; ii++) {
        err = cudlaMemRegister(devHandle,
                               (uint64_t* )(inputBufferGPU[ii]),
                               inputTensorDesc[ii].size,
                               &(inputBufferRegisteredPtr[ii]),
                               0);
        if (err != cudlaSuccess) {
            DPRINTF("Error in registering input memory = %d\n", err);
            cleanUp(&resourceList);
            return 1;
        }
    }

    for (uint32_t ii = 0; ii < numOutputTensors; ii++) {
        err = cudlaMemRegister(devHandle,
                               (uint64_t* )(outputBufferGPU[ii]),
                               outputTensorDesc[ii].size,
                               &(outputBufferRegisteredPtr[ii]),
                               0);
        if (err != cudlaSuccess) {
            DPRINTF("Error in registering output memory = %d\n", err);
            cleanUp(&resourceList);
            return 1;
        }
    }

    for (uint32_t ii = 0; ii < numOutputTaskStatistics; ii++) {
        err = cudlaMemRegister(devHandle,
                               (uint64_t* )(outputTaskStatisticsGPU[ii]),
                               outputTaskStatisticsDesc[ii].size,
                               &(outputTaskStatisticsRegisteredPtr[ii]),
                               CUDLA_TASK_STATISTICS);
        if (err != cudlaSuccess) {
            DPRINTF("Error in registering statistics output memory = %d\n", err);
            cleanUp(&resourceList);
            return 1;
        }
    }

    DPRINTF("ALL MEMORY REGISTERED SUCCESSFULLY\n");

    // Copy data from CPU buffers to GPU buffers.
    for (uint32_t ii = 0; ii < numInputTensors; ii++) {
        result = cudaMemcpyAsync(inputBufferGPU[ii], inputBuffer[ii], inputTensorDesc[ii].size, cudaMemcpyHostToDevice, stream);
        if (result != cudaSuccess) {
            DPRINTF("Error in enqueueing memcpy for input\n");
            cleanUp(&resourceList);
            return 1;
        }
    }

    for (uint32_t ii = 0; ii < numOutputTensors; ii++) {
        result = cudaMemsetAsync(outputBufferGPU[ii], 0, outputTensorDesc[ii].size, stream);
        if (result != cudaSuccess) {
            DPRINTF("Error in enqueueing memset for output\n");
            cleanUp(&resourceList);
            return 1;
        }
    }

    for (uint32_t ii = 0; ii < numOutputTaskStatistics; ii++) {
        result = cudaMemsetAsync(outputTaskStatisticsGPU[ii], 0, outputTaskStatisticsDesc[ii].size, stream);
        if (result != cudaSuccess) {
            DPRINTF("Error in enqueueing memset for statistics output\n");
            cleanUp(&resourceList);
            return 1;
        }
    }

    uint64_t *outputStatisticsBufferRegisteredPtr[numOutputTensors + numOutputTaskStatistics] = {0};
    uint32_t index = 0;
    for (; index < numOutputTensors ; index++) {
        outputStatisticsBufferRegisteredPtr[index] = ((outputBufferRegisteredPtr[index]));
    }

    for (uint32_t jj=0; jj < numOutputTaskStatistics ; jj++) {
        outputStatisticsBufferRegisteredPtr[index++] = ((outputTaskStatisticsRegisteredPtr[jj]));
    }

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
    task.inputTensor = inputBufferRegisteredPtr;
    task.waitEvents = NULL;
    task.signalEvents = NULL;

    err = cudlaSubmitTask(devHandle, &task, 1, stream, 0);
    if (err != cudlaSuccess) {
        DPRINTF("no of output tensor %u \n",(task.numOutputTensors));
        DPRINTF("Error in submitting task\n");
        cleanUp(&resourceList);
        return 1;
    }
    DPRINTF("SUBMIT IS DONE !!!\n");

    result = cudaStreamSynchronize(stream);
    if (result != cudaSuccess) {
        DPRINTF("Error in synchronizing stream = %s\n", cudaGetErrorName(result));
        cleanUp(&resourceList);
        return 1;
    }

    // Wait for stream operations to finish and bring output buffer to CPU.
    for (uint32_t ii = 0; ii < numOutputTensors; ii++) {
        result = cudaMemcpyAsync(outputBuffer[ii], outputBufferGPU[ii],
                                 outputTensorDesc[ii].size, cudaMemcpyDeviceToHost, stream);
        if (result != cudaSuccess) {
            DPRINTF("Error in bringing result back to CPU\n");
            cleanUp(&resourceList);
            return 1;
        }
    }

    result = cudaStreamSynchronize(stream);
    if (result != cudaSuccess) {
        DPRINTF("Error in synchronizing stream\n");
        cleanUp(&resourceList);
        return 1;
    }

    if(statSupport == 1) {
        // copy statistics data to cpu
        for (uint32_t ii = 0; ii < numOutputTaskStatistics; ii++) {
            result = cudaMemcpyAsync(statisticsOutputBuffer[ii], outputTaskStatisticsGPU[ii],
                                     outputTaskStatisticsDesc[ii].size, cudaMemcpyDeviceToHost, stream);
            if (result != cudaSuccess) {
                DPRINTF("Error in bringing result back to CPU\n");
                cleanUp(&resourceList);
                return 1;
            }
        }

        result = cudaStreamSynchronize(stream);
        if (result != cudaSuccess) {
            DPRINTF("Error in synchronizing stream\n");
            cleanUp(&resourceList);
            return 1;
        }

        // To get the last index of the filename prefix in which statistics will be dumped
        uint32_t index = 0;
        if (argc == 5) {
            while(argv[4][index]!='\0') {
                index++;
            }
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
                                 (inputBufferRegisteredPtr[ii]));
        if (err != cudlaSuccess) {
            DPRINTF("Error in registering input memory = %d\n", err);
            cleanUp(&resourceList);
            return 1;
        }
    }

    for (uint32_t ii = 0; ii < numOutputTensors; ii++) {
        err = cudlaMemUnregister(devHandle,
                                 (outputBufferRegisteredPtr[ii]));
        if (err != cudlaSuccess) {
            DPRINTF("Error in registering output memory = %d\n", err);
            cleanUp(&resourceList);
            return 1;
        }
    }

    for (uint32_t ii = 0; ii < numOutputTaskStatistics; ii++) {
        err = cudlaMemUnregister(devHandle,
                                 (outputTaskStatisticsRegisteredPtr[ii]));
        if (err != cudlaSuccess) {
            DPRINTF("Error in registering output memory = %d\n", err);
            cleanUp(&resourceList);
            return 1;
        }
    }

    DPRINTF("ALL MEMORY UNREGISTERED SUCCESSFULLY\n");

    result = cudaStreamDestroy(stream);
    if (result != cudaSuccess) {
        errPtr = cudaGetErrorName(result);
        DPRINTF("Error in destroying cuda stream = %s\n", errPtr);
        cleanUp(&resourceList);
        return 1;
    }

    resourceList.stream = NULL;

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

    DPRINTF("cuDLALayerwiseStatsHybrid DONE !!!\n");

    return 0;
}
