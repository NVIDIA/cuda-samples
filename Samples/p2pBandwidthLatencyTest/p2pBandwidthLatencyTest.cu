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

#include <cstdio>
#include <vector>

#include <helper_cuda.h>
#include <helper_timer.h>

using namespace std;

const char *sSampleName = "P2P (Peer-to-Peer) GPU Bandwidth Latency Test";

typedef enum {
  P2P_WRITE = 0,
  P2P_READ = 1,
} P2PDataTransfer;

typedef enum {
  CE = 0,
  SM = 1,
} P2PEngine;

P2PEngine p2p_mechanism = CE;  // By default use Copy Engine

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }
__global__ void delay(volatile int *flag,
                      unsigned long long timeout_clocks = 10000000) {
  // Wait until the application notifies us that it has completed queuing up the
  // experiment, or timeout and exit, allowing the application to make progress
  long long int start_clock, sample_clock;
  start_clock = clock64();

  while (!*flag) {
    sample_clock = clock64();

    if (sample_clock - start_clock > timeout_clocks) {
      break;
    }
  }
}

// This kernel is for demonstration purposes only, not a performant kernel for
// p2p transfers.
__global__ void copyp2p(int4 *__restrict__ dest, int4 const *__restrict__ src,
                        size_t num_elems) {
  size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t gridSize = blockDim.x * gridDim.x;

#pragma unroll(5)
  for (size_t i = globalId; i < num_elems; i += gridSize) {
    dest[i] = src[i];
  }
}

///////////////////////////////////////////////////////////////////////////
// Print help screen
///////////////////////////////////////////////////////////////////////////
void printHelp(void) {
  printf("Usage:  p2pBandwidthLatencyTest [OPTION]...\n");
  printf("Tests bandwidth/latency of GPU pairs using P2P and without P2P\n");
  printf("\n");

  printf("Options:\n");
  printf("--help\t\tDisplay this help menu\n");
  printf(
      "--p2p_read\tUse P2P reads for data transfers between GPU pairs and show "
      "corresponding results.\n \t\tDefault used is P2P write operation.\n");
  printf("--sm_copy\t\tUse SM intiated p2p transfers instead of Copy Engine\n");
}

void checkP2Paccess(int numGPUs) {
  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i);
    cudaCheckError();

    for (int j = 0; j < numGPUs; j++) {
      int access;
      if (i != j) {
        cudaDeviceCanAccessPeer(&access, i, j);
        cudaCheckError();
        printf("Device=%d %s Access Peer Device=%d\n", i,
               access ? "CAN" : "CANNOT", j);
      }
    }
  }
  printf(
      "\n***NOTE: In case a device doesn't have P2P access to other one, it "
      "falls back to normal memcopy procedure.\nSo you can see lesser "
      "Bandwidth (GB/s) and unstable Latency (us) in those cases.\n\n");
}

void performP2PCopy(int *dest, int destDevice, int *src, int srcDevice,
                    int num_elems, int repeat, bool p2paccess,
                    cudaStream_t streamToRun) {
  int blockSize = 0;
  int numBlocks = 0;

  cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, copyp2p);
  cudaCheckError();

  if (p2p_mechanism == SM && p2paccess) {
    for (int r = 0; r < repeat; r++) {
      copyp2p<<<numBlocks, blockSize, 0, streamToRun>>>(
          (int4 *)dest, (int4 *)src, num_elems / 4);
    }
  } else {
    for (int r = 0; r < repeat; r++) {
      cudaMemcpyPeerAsync(dest, destDevice, src, srcDevice,
                          sizeof(int) * num_elems, streamToRun);
    }
  }
}

void outputBandwidthMatrix(int numGPUs, bool p2p, P2PDataTransfer p2p_method) {
  int numElems = 10000000;
  int repeat = 5;
  volatile int *flag = NULL;
  vector<int *> buffers(numGPUs);
  vector<int *> buffersD2D(numGPUs);  // buffer for D2D, that is, intra-GPU copy
  vector<cudaEvent_t> start(numGPUs);
  vector<cudaEvent_t> stop(numGPUs);
  vector<cudaStream_t> stream(numGPUs);

  cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable);
  cudaCheckError();

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking);
    cudaMalloc(&buffers[d], numElems * sizeof(int));
    cudaCheckError();
    cudaMemset(buffers[d], 0, numElems * sizeof(int));
    cudaCheckError();
    cudaMalloc(&buffersD2D[d], numElems * sizeof(int));
    cudaCheckError();
    cudaMemset(buffersD2D[d], 0, numElems * sizeof(int));
    cudaCheckError();
    cudaEventCreate(&start[d]);
    cudaCheckError();
    cudaEventCreate(&stop[d]);
    cudaCheckError();
  }

  vector<double> bandwidthMatrix(numGPUs * numGPUs);

  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i);

    for (int j = 0; j < numGPUs; j++) {
      int access = 0;
      if (p2p) {
        cudaDeviceCanAccessPeer(&access, i, j);
        if (access) {
          cudaDeviceEnablePeerAccess(j, 0);
          cudaCheckError();
          cudaSetDevice(j);
          cudaCheckError();
          cudaDeviceEnablePeerAccess(i, 0);
          cudaCheckError();
          cudaSetDevice(i);
          cudaCheckError();
        }
      }

      cudaStreamSynchronize(stream[i]);
      cudaCheckError();

      // Block the stream until all the work is queued up
      // DANGER! - cudaMemcpy*Async may infinitely block waiting for
      // room to push the operation, so keep the number of repeatitions
      // relatively low.  Higher repeatitions will cause the delay kernel
      // to timeout and lead to unstable results.
      *flag = 0;
      delay<<<1, 1, 0, stream[i]>>>(flag);
      cudaCheckError();
      cudaEventRecord(start[i], stream[i]);
      cudaCheckError();

      if (i == j) {
        // Perform intra-GPU, D2D copies
        performP2PCopy(buffers[i], i, buffersD2D[i], i, numElems, repeat,
                       access, stream[i]);

      } else {
        if (p2p_method == P2P_WRITE) {
          performP2PCopy(buffers[j], j, buffers[i], i, numElems, repeat, access,
                         stream[i]);
        } else {
          performP2PCopy(buffers[i], i, buffers[j], j, numElems, repeat, access,
                         stream[i]);
        }
      }

      cudaEventRecord(stop[i], stream[i]);
      cudaCheckError();

      // Release the queued events
      *flag = 1;
      cudaStreamSynchronize(stream[i]);
      cudaCheckError();

      float time_ms;
      cudaEventElapsedTime(&time_ms, start[i], stop[i]);
      double time_s = time_ms / 1e3;

      double gb = numElems * sizeof(int) * repeat / (double)1e9;
      if (i == j) {
        gb *= 2;  // must count both the read and the write here
      }
      bandwidthMatrix[i * numGPUs + j] = gb / time_s;
      if (p2p && access) {
        cudaDeviceDisablePeerAccess(j);
        cudaSetDevice(j);
        cudaDeviceDisablePeerAccess(i);
        cudaSetDevice(i);
        cudaCheckError();
      }
    }
  }

  printf("   D\\D");

  for (int j = 0; j < numGPUs; j++) {
    printf("%6d ", j);
  }

  printf("\n");

  for (int i = 0; i < numGPUs; i++) {
    printf("%6d ", i);

    for (int j = 0; j < numGPUs; j++) {
      printf("%6.02f ", bandwidthMatrix[i * numGPUs + j]);
    }

    printf("\n");
  }

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaFree(buffers[d]);
    cudaFree(buffersD2D[d]);
    cudaCheckError();
    cudaEventDestroy(start[d]);
    cudaCheckError();
    cudaEventDestroy(stop[d]);
    cudaCheckError();
    cudaStreamDestroy(stream[d]);
    cudaCheckError();
  }

  cudaFreeHost((void *)flag);
  cudaCheckError();
}

void outputBidirectionalBandwidthMatrix(int numGPUs, bool p2p) {
  int numElems = 10000000;
  int repeat = 5;
  volatile int *flag = NULL;
  vector<int *> buffers(numGPUs);
  vector<int *> buffersD2D(numGPUs);
  vector<cudaEvent_t> start(numGPUs);
  vector<cudaEvent_t> stop(numGPUs);
  vector<cudaStream_t> stream0(numGPUs);
  vector<cudaStream_t> stream1(numGPUs);

  cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable);
  cudaCheckError();

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaMalloc(&buffers[d], numElems * sizeof(int));
    cudaMemset(buffers[d], 0, numElems * sizeof(int));
    cudaMalloc(&buffersD2D[d], numElems * sizeof(int));
    cudaMemset(buffersD2D[d], 0, numElems * sizeof(int));
    cudaCheckError();
    cudaEventCreate(&start[d]);
    cudaCheckError();
    cudaEventCreate(&stop[d]);
    cudaCheckError();
    cudaStreamCreateWithFlags(&stream0[d], cudaStreamNonBlocking);
    cudaCheckError();
    cudaStreamCreateWithFlags(&stream1[d], cudaStreamNonBlocking);
    cudaCheckError();
  }

  vector<double> bandwidthMatrix(numGPUs * numGPUs);

  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i);

    for (int j = 0; j < numGPUs; j++) {
      int access = 0;
      if (p2p) {
        cudaDeviceCanAccessPeer(&access, i, j);
        if (access) {
          cudaSetDevice(i);
          cudaDeviceEnablePeerAccess(j, 0);
          cudaCheckError();
          cudaSetDevice(j);
          cudaDeviceEnablePeerAccess(i, 0);
          cudaCheckError();
        }
      }

      cudaSetDevice(i);
      cudaStreamSynchronize(stream0[i]);
      cudaStreamSynchronize(stream1[j]);
      cudaCheckError();

      // Block the stream until all the work is queued up
      // DANGER! - cudaMemcpy*Async may infinitely block waiting for
      // room to push the operation, so keep the number of repeatitions
      // relatively low.  Higher repeatitions will cause the delay kernel
      // to timeout and lead to unstable results.
      *flag = 0;
      cudaSetDevice(i);
      // No need to block stream1 since it'll be blocked on stream0's event
      delay<<<1, 1, 0, stream0[i]>>>(flag);
      cudaCheckError();

      // Force stream1 not to start until stream0 does, in order to ensure
      // the events on stream0 fully encompass the time needed for all
      // operations
      cudaEventRecord(start[i], stream0[i]);
      cudaStreamWaitEvent(stream1[j], start[i], 0);

      if (i == j) {
        // For intra-GPU perform 2 memcopies buffersD2D <-> buffers
        performP2PCopy(buffers[i], i, buffersD2D[i], i, numElems, repeat,
                       access, stream0[i]);
        performP2PCopy(buffersD2D[i], i, buffers[i], i, numElems, repeat,
                       access, stream1[i]);
      } else {
        if (access && p2p_mechanism == SM) {
          cudaSetDevice(j);
        }
        performP2PCopy(buffers[i], i, buffers[j], j, numElems, repeat, access,
                       stream1[j]);
        if (access && p2p_mechanism == SM) {
          cudaSetDevice(i);
        }
        performP2PCopy(buffers[j], j, buffers[i], i, numElems, repeat, access,
                       stream0[i]);
      }

      // Notify stream0 that stream1 is complete and record the time of
      // the total transaction
      cudaEventRecord(stop[j], stream1[j]);
      cudaStreamWaitEvent(stream0[i], stop[j], 0);
      cudaEventRecord(stop[i], stream0[i]);

      // Release the queued operations
      *flag = 1;
      cudaStreamSynchronize(stream0[i]);
      cudaStreamSynchronize(stream1[j]);
      cudaCheckError();

      float time_ms;
      cudaEventElapsedTime(&time_ms, start[i], stop[i]);
      double time_s = time_ms / 1e3;

      double gb = 2.0 * numElems * sizeof(int) * repeat / (double)1e9;
      if (i == j) {
        gb *= 2;  // must count both the read and the write here
      }
      bandwidthMatrix[i * numGPUs + j] = gb / time_s;
      if (p2p && access) {
        cudaSetDevice(i);
        cudaDeviceDisablePeerAccess(j);
        cudaSetDevice(j);
        cudaDeviceDisablePeerAccess(i);
      }
    }
  }

  printf("   D\\D");

  for (int j = 0; j < numGPUs; j++) {
    printf("%6d ", j);
  }

  printf("\n");

  for (int i = 0; i < numGPUs; i++) {
    printf("%6d ", i);

    for (int j = 0; j < numGPUs; j++) {
      printf("%6.02f ", bandwidthMatrix[i * numGPUs + j]);
    }

    printf("\n");
  }

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaFree(buffers[d]);
    cudaFree(buffersD2D[d]);
    cudaCheckError();
    cudaEventDestroy(start[d]);
    cudaCheckError();
    cudaEventDestroy(stop[d]);
    cudaCheckError();
    cudaStreamDestroy(stream0[d]);
    cudaCheckError();
    cudaStreamDestroy(stream1[d]);
    cudaCheckError();
  }

  cudaFreeHost((void *)flag);
  cudaCheckError();
}

void outputLatencyMatrix(int numGPUs, bool p2p, P2PDataTransfer p2p_method) {
  int repeat = 100;
  int numElems = 4;  // perform 1-int4 transfer.
  volatile int *flag = NULL;
  StopWatchInterface *stopWatch = NULL;
  vector<int *> buffers(numGPUs);
  vector<int *> buffersD2D(numGPUs);  // buffer for D2D, that is, intra-GPU copy
  vector<cudaStream_t> stream(numGPUs);
  vector<cudaEvent_t> start(numGPUs);
  vector<cudaEvent_t> stop(numGPUs);

  cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable);
  cudaCheckError();

  if (!sdkCreateTimer(&stopWatch)) {
    printf("Failed to create stop watch\n");
    exit(EXIT_FAILURE);
  }
  sdkStartTimer(&stopWatch);

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking);
    cudaMalloc(&buffers[d], sizeof(int) * numElems);
    cudaMemset(buffers[d], 0, sizeof(int) * numElems);
    cudaMalloc(&buffersD2D[d], sizeof(int) * numElems);
    cudaMemset(buffersD2D[d], 0, sizeof(int) * numElems);
    cudaCheckError();
    cudaEventCreate(&start[d]);
    cudaCheckError();
    cudaEventCreate(&stop[d]);
    cudaCheckError();
  }

  vector<double> gpuLatencyMatrix(numGPUs * numGPUs);
  vector<double> cpuLatencyMatrix(numGPUs * numGPUs);

  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i);

    for (int j = 0; j < numGPUs; j++) {
      int access = 0;
      if (p2p) {
        cudaDeviceCanAccessPeer(&access, i, j);
        if (access) {
          cudaDeviceEnablePeerAccess(j, 0);
          cudaCheckError();
          cudaSetDevice(j);
          cudaDeviceEnablePeerAccess(i, 0);
          cudaSetDevice(i);
          cudaCheckError();
        }
      }
      cudaStreamSynchronize(stream[i]);
      cudaCheckError();

      // Block the stream until all the work is queued up
      // DANGER! - cudaMemcpy*Async may infinitely block waiting for
      // room to push the operation, so keep the number of repeatitions
      // relatively low.  Higher repeatitions will cause the delay kernel
      // to timeout and lead to unstable results.
      *flag = 0;
      delay<<<1, 1, 0, stream[i]>>>(flag);
      cudaCheckError();
      cudaEventRecord(start[i], stream[i]);

      sdkResetTimer(&stopWatch);
      if (i == j) {
        // Perform intra-GPU, D2D copies
        performP2PCopy(buffers[i], i, buffersD2D[i], i, numElems, repeat,
                       access, stream[i]);
      } else {
        if (p2p_method == P2P_WRITE) {
          performP2PCopy(buffers[j], j, buffers[i], i, numElems, repeat, access,
                         stream[i]);
        } else {
          performP2PCopy(buffers[i], i, buffers[j], j, numElems, repeat, access,
                         stream[i]);
        }
      }
      float cpu_time_ms = sdkGetTimerValue(&stopWatch);

      cudaEventRecord(stop[i], stream[i]);
      // Now that the work has been queued up, release the stream
      *flag = 1;
      cudaStreamSynchronize(stream[i]);
      cudaCheckError();

      float gpu_time_ms;
      cudaEventElapsedTime(&gpu_time_ms, start[i], stop[i]);

      gpuLatencyMatrix[i * numGPUs + j] = gpu_time_ms * 1e3 / repeat;
      cpuLatencyMatrix[i * numGPUs + j] = cpu_time_ms * 1e3 / repeat;
      if (p2p && access) {
        cudaDeviceDisablePeerAccess(j);
        cudaSetDevice(j);
        cudaDeviceDisablePeerAccess(i);
        cudaSetDevice(i);
        cudaCheckError();
      }
    }
  }

  printf("   GPU");

  for (int j = 0; j < numGPUs; j++) {
    printf("%6d ", j);
  }

  printf("\n");

  for (int i = 0; i < numGPUs; i++) {
    printf("%6d ", i);

    for (int j = 0; j < numGPUs; j++) {
      printf("%6.02f ", gpuLatencyMatrix[i * numGPUs + j]);
    }

    printf("\n");
  }

  printf("\n   CPU");

  for (int j = 0; j < numGPUs; j++) {
    printf("%6d ", j);
  }

  printf("\n");

  for (int i = 0; i < numGPUs; i++) {
    printf("%6d ", i);

    for (int j = 0; j < numGPUs; j++) {
      printf("%6.02f ", cpuLatencyMatrix[i * numGPUs + j]);
    }

    printf("\n");
  }

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaFree(buffers[d]);
    cudaFree(buffersD2D[d]);
    cudaCheckError();
    cudaEventDestroy(start[d]);
    cudaCheckError();
    cudaEventDestroy(stop[d]);
    cudaCheckError();
    cudaStreamDestroy(stream[d]);
    cudaCheckError();
  }

  sdkDeleteTimer(&stopWatch);

  cudaFreeHost((void *)flag);
  cudaCheckError();
}

int main(int argc, char **argv) {
  int numGPUs;
  P2PDataTransfer p2p_method = P2P_WRITE;

  cudaGetDeviceCount(&numGPUs);
  cudaCheckError();

  // process command line args
  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printHelp();
    return 0;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "p2p_read")) {
    p2p_method = P2P_READ;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "sm_copy")) {
    p2p_mechanism = SM;
  }

  printf("[%s]\n", sSampleName);

  // output devices
  for (int i = 0; i < numGPUs; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cudaCheckError();
    printf("Device: %d, %s, pciBusID: %x, pciDeviceID: %x, pciDomainID:%x\n", i,
           prop.name, prop.pciBusID, prop.pciDeviceID, prop.pciDomainID);
  }

  checkP2Paccess(numGPUs);

  // Check peer-to-peer connectivity
  printf("P2P Connectivity Matrix\n");
  printf("     D\\D");

  for (int j = 0; j < numGPUs; j++) {
    printf("%6d", j);
  }
  printf("\n");

  for (int i = 0; i < numGPUs; i++) {
    printf("%6d\t", i);
    for (int j = 0; j < numGPUs; j++) {
      if (i != j) {
        int access;
        cudaDeviceCanAccessPeer(&access, i, j);
        cudaCheckError();
        printf("%6d", (access) ? 1 : 0);
      } else {
        printf("%6d", 1);
      }
    }
    printf("\n");
  }

  printf("Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)\n");
  outputBandwidthMatrix(numGPUs, false, P2P_WRITE);
  printf("Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)\n");
  outputBandwidthMatrix(numGPUs, true, P2P_WRITE);
  if (p2p_method == P2P_READ) {
    printf("Unidirectional P2P=Enabled Bandwidth (P2P Reads) Matrix (GB/s)\n");
    outputBandwidthMatrix(numGPUs, true, p2p_method);
  }
  printf("Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)\n");
  outputBidirectionalBandwidthMatrix(numGPUs, false);
  printf("Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)\n");
  outputBidirectionalBandwidthMatrix(numGPUs, true);

  printf("P2P=Disabled Latency Matrix (us)\n");
  outputLatencyMatrix(numGPUs, false, P2P_WRITE);
  printf("P2P=Enabled Latency (P2P Writes) Matrix (us)\n");
  outputLatencyMatrix(numGPUs, true, P2P_WRITE);
  if (p2p_method == P2P_READ) {
    printf("P2P=Enabled Latency (P2P Reads) Matrix (us)\n");
    outputLatencyMatrix(numGPUs, true, p2p_method);
  }

  printf(
      "\nNOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n");

  exit(EXIT_SUCCESS);
}
