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
#include "cuda_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <fstream>
#include <sstream>

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
  unsigned char* inputBuffer;
  unsigned char* outputBuffer;
  void* inputBufferGPU;
  void* outputBufferGPU;
  cudlaModuleTensorDescriptor* inputTensorDesc;
  cudlaModuleTensorDescriptor* outputTensorDesc;
} ResourceList;

void cleanUp(ResourceList* resourceList);

void cleanUp(ResourceList* resourceList) {
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

  if (resourceList->inputBufferGPU != 0) {
    cudaFree(resourceList->inputBufferGPU);
    resourceList->inputBufferGPU = 0;
  }
  if (resourceList->outputBufferGPU != 0) {
    cudaFree(resourceList->outputBufferGPU);
    resourceList->outputBufferGPU = 0;
  }

  if (resourceList->inputBuffer != NULL) {
    free(resourceList->inputBuffer);
    resourceList->inputBuffer = NULL;
  }
  if (resourceList->outputBuffer != NULL) {
    free(resourceList->outputBuffer);
    resourceList->outputBuffer = NULL;
  }

  if (resourceList->stream != NULL) {
    cudaStreamDestroy(resourceList->stream);
    resourceList->stream = NULL;
  }
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

  cudaStream_t stream;
  cudaError_t result;
  const char* errPtr = NULL;

  ResourceList resourceList;

  memset(&resourceList, 0x00, sizeof(ResourceList));

  if (argc != 2) {
    DPRINTF("Usage : ./cuDLAErrorReporting <loadable>\n");
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

  memset(inputBuffer, 0x01, inputTensorDesc[0].size);
  memset(outputBuffer, 0x00, outputTensorDesc[0].size);

  // Allocate memory on GPU.
  void* inputBufferGPU;
  void* outputBufferGPU;
  result = cudaMalloc(&inputBufferGPU, inputTensorDesc[0].size);
  if (result != cudaSuccess) {
    DPRINTF("Error in allocating input memory on GPU\n");
    cleanUp(&resourceList);
    return 1;
  }

  resourceList.inputBufferGPU = inputBufferGPU;

  result = cudaMalloc(&outputBufferGPU, outputTensorDesc[0].size);
  if (result != cudaSuccess) {
    DPRINTF("Error in allocating output memory on GPU\n");
    cleanUp(&resourceList);
    return 1;
  }

  resourceList.outputBufferGPU = outputBufferGPU;

  // Register the CUDA-allocated buffers.
  uint64_t* inputBufferRegisteredPtr = NULL;
  uint64_t* outputBufferRegisteredPtr = NULL;

  err = cudlaMemRegister(devHandle, (uint64_t*)inputBufferGPU,
                         inputTensorDesc[0].size, &inputBufferRegisteredPtr, 0);
  if (err != cudlaSuccess) {
    DPRINTF("Error in registering input memory = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }

  err =
      cudlaMemRegister(devHandle, (uint64_t*)outputBufferGPU,
                       outputTensorDesc[0].size, &outputBufferRegisteredPtr, 0);
  if (err != cudlaSuccess) {
    DPRINTF("Error in registering output memory = %d\n", err);
    cleanUp(&resourceList);
    return 1;
  }
  DPRINTF("ALL MEMORY REGISTERED SUCCESSFULLY\n");

  // Copy data from CPU buffers to GPU buffers.
  result = cudaMemcpyAsync(inputBufferGPU, inputBuffer, inputTensorDesc[0].size,
                           cudaMemcpyHostToDevice, stream);
  if (result != cudaSuccess) {
    DPRINTF("Error in enqueueing memcpy for input\n");
    cleanUp(&resourceList);
    return 1;
  }
  result =
      cudaMemsetAsync(outputBufferGPU, 0, outputTensorDesc[0].size, stream);
  if (result != cudaSuccess) {
    DPRINTF("Error in enqueueing memset for output\n");
    cleanUp(&resourceList);
    return 1;
  }

  // Enqueue a cuDLA task.
  cudlaTask task;
  task.moduleHandle = moduleHandle;
  task.outputTensor = &outputBufferRegisteredPtr;
  task.numOutputTensors = 1;
  task.numInputTensors = 1;
  task.inputTensor = &inputBufferRegisteredPtr;
  task.waitEvents = NULL;
  task.signalEvents = NULL;
  err = cudlaSubmitTask(devHandle, &task, 1, stream, 0);
  if (err != cudlaSuccess) {
    DPRINTF("Error in submitting task\n");
    cleanUp(&resourceList);
    return 1;
  }
  DPRINTF("SUBMIT IS DONE !!!\n");

  // Wait for stream operations to finish and bring output buffer to CPU.
  result =
      cudaMemcpyAsync(outputBuffer, outputBufferGPU, outputTensorDesc[0].size,
                      cudaMemcpyDeviceToHost, stream);
  if (result != cudaSuccess) {
    if (result != cudaErrorExternalDevice) {
      DPRINTF("Error in bringing result back to CPU\n");
      cleanUp(&resourceList);
      return 1;
    } else {
      cudlaStatus hwStatus = cudlaGetLastError(devHandle);
      if (hwStatus != cudlaSuccess) {
        DPRINTF("Asynchronous error in HW = %u\n", hwStatus);
      }
    }
  }

  result = cudaStreamSynchronize(stream);
  if (result != cudaSuccess) {
    DPRINTF("Error in synchronizing stream = %s\n", cudaGetErrorName(result));

    if (result == cudaErrorExternalDevice) {
      cudlaStatus hwStatus = cudlaGetLastError(devHandle);
      if (hwStatus != cudlaSuccess) {
        DPRINTF("Asynchronous error in HW = %u\n", hwStatus);
      }
    }
  }

  cleanUp(&resourceList);

  DPRINTF("cuDLAErrorReporting DONE !!!\n");

  return 0;
}
