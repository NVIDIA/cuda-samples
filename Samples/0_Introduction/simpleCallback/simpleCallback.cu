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

/*
 * This sample implements multi-threaded heterogeneous computing workloads with
 * the new CPU callbacks for CUDA streams and events introduced with CUDA 5.0.
 * Together with the thread safety of the CUDA API implementing heterogeneous
 * workloads that float between CPU threads and GPUs has become simple and
 * efficient.
 *
 * The workloads in the sample follow the form CPU preprocess -> GPU process ->
 * CPU postprocess.
 * Each CPU processing step is handled by its own dedicated thread. GPU
 * workloads are sent to all available GPUs in the system.
 *
 */

// System includes
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "multithreading.h"

const int N_workloads = 8;
const int N_elements_per_workload = 100000;

CUTBarrier thread_barrier;

void CUDART_CB myStreamCallback(cudaStream_t event, cudaError_t status,
                                void *data);

struct heterogeneous_workload {
  int id;
  int cudaDeviceID;

  int *h_data;
  int *d_data;
  cudaStream_t stream;

  bool success;
};

__global__ void incKernel(int *data, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) data[i]++;
}

CUT_THREADPROC launch(void *void_arg) {
  heterogeneous_workload *workload = (heterogeneous_workload *)void_arg;

  // Select GPU for this CPU thread
  checkCudaErrors(cudaSetDevice(workload->cudaDeviceID));

  // Allocate Resources
  checkCudaErrors(cudaStreamCreate(&workload->stream));
  checkCudaErrors(
      cudaMalloc(&workload->d_data, N_elements_per_workload * sizeof(int)));
  checkCudaErrors(cudaHostAlloc(&workload->h_data,
                                N_elements_per_workload * sizeof(int),
                                cudaHostAllocPortable));

  // CPU thread generates data
  for (int i = 0; i < N_elements_per_workload; ++i) {
    workload->h_data[i] = workload->id + i;
  }

  // Schedule work for GPU in CUDA stream without blocking the CPU thread
  // Note: Dedicated streams enable concurrent execution of workloads on the GPU
  dim3 block(512);
  dim3 grid((N_elements_per_workload + block.x - 1) / block.x);

  checkCudaErrors(cudaMemcpyAsync(workload->d_data, workload->h_data,
                                  N_elements_per_workload * sizeof(int),
                                  cudaMemcpyHostToDevice, workload->stream));
  incKernel<<<grid, block, 0, workload->stream>>>(workload->d_data,
                                                  N_elements_per_workload);
  checkCudaErrors(cudaMemcpyAsync(workload->h_data, workload->d_data,
                                  N_elements_per_workload * sizeof(int),
                                  cudaMemcpyDeviceToHost, workload->stream));

  // New in CUDA 5.0: Add a CPU callback which is called once all currently
  // pending operations in the CUDA stream have finished
  checkCudaErrors(
      cudaStreamAddCallback(workload->stream, myStreamCallback, workload, 0));

  CUT_THREADEND;
  // CPU thread end of life, GPU continues to process data...
}

CUT_THREADPROC postprocess(void *void_arg) {
  heterogeneous_workload *workload = (heterogeneous_workload *)void_arg;
  // ... GPU is done with processing, continue on new CPU thread...

  // Select GPU for this CPU thread
  checkCudaErrors(cudaSetDevice(workload->cudaDeviceID));

  // CPU thread consumes results from GPU
  workload->success = true;

  for (int i = 0; i < N_workloads; ++i) {
    workload->success &= workload->h_data[i] == i + workload->id + 1;
  }

  // Free Resources
  checkCudaErrors(cudaFree(workload->d_data));
  checkCudaErrors(cudaFreeHost(workload->h_data));
  checkCudaErrors(cudaStreamDestroy(workload->stream));

  // Signal the end of the heterogeneous workload to main thread
  cutIncrementBarrier(&thread_barrier);

  CUT_THREADEND;
}

void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status,
                                void *data) {
  // Check status of GPU after stream operations are done
  checkCudaErrors(status);

  // Spawn new CPU worker thread and continue processing on the CPU
  cutStartThread(postprocess, data);
}

int main(int argc, char **argv) {
  int N_gpus, max_gpus = 0;
  int gpuInfo[32];  // assume a maximum of 32 GPUs in a system configuration

  printf("Starting simpleCallback\n");

  checkCudaErrors(cudaGetDeviceCount(&N_gpus));
  printf("Found %d CUDA capable GPUs\n", N_gpus);

  if (N_gpus > 32) {
    printf("simpleCallback only supports 32 GPU(s)\n");
  }

  for (int devid = 0; devid < N_gpus; devid++) {
    int SMversion;
    cudaDeviceProp deviceProp;
    cudaSetDevice(devid);
    cudaGetDeviceProperties(&deviceProp, devid);
    SMversion = deviceProp.major << 4 + deviceProp.minor;
    printf("GPU[%d] %s supports SM %d.%d", devid, deviceProp.name,
           deviceProp.major, deviceProp.minor);
    printf(", %s GPU Callback Functions\n",
           (SMversion >= 0x11) ? "capable" : "NOT capable");

    if (SMversion >= 0x11) {
      gpuInfo[max_gpus++] = devid;
    }
  }

  printf("%d GPUs available to run Callback Functions\n", max_gpus);

  heterogeneous_workload *workloads;
  workloads = (heterogeneous_workload *)malloc(N_workloads *
                                               sizeof(heterogeneous_workload));
  ;
  thread_barrier = cutCreateBarrier(N_workloads);

  // Main thread spawns a CPU worker thread for each heterogeneous workload
  printf("Starting %d heterogeneous computing workloads\n", N_workloads);

  for (int i = 0; i < N_workloads; ++i) {
    workloads[i].id = i;
    workloads[i].cudaDeviceID = gpuInfo[i % max_gpus];  // i % N_gpus;

    cutStartThread(launch, &workloads[i]);
  }

  // Sleep until all workloads have finished
  cutWaitForBarrier(&thread_barrier);
  printf("Total of %d workloads finished:\n", N_workloads);

  bool success = true;

  for (int i = 0; i < N_workloads; ++i) {
    success &= workloads[i].success;
  }

  printf("%s\n", success ? "Success" : "Failure");

  free(workloads);

  exit(success ? EXIT_SUCCESS : EXIT_FAILURE);
}
