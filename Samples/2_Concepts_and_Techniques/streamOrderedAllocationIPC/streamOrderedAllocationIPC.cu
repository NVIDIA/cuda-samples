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
 * This sample demonstrates Inter Process Communication
 * using one process per GPU for computation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda.h>
#define CUDA_DRIVER_API 1
#include "helper_cuda.h"
#include "helper_cuda_drvapi.h"
#include "helper_multiprocess.h"

static const char shmName[] = "streamOrderedAllocationIPCshm";
static const char ipcName[] = "streamOrderedAllocationIPC_pipe";
// For direct NVLINK and PCI-E peers, at max 8 simultaneous peers are allowed
// For NVSWITCH connected peers like DGX-2, simultaneous peers are not limited
// in the same way.
#define MAX_DEVICES (32)
#define DATA_SIZE (64ULL << 20ULL)  // 64MB

#if defined(__linux__)
#define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define cpu_atomic_add32(a, x) InterlockedAdd((volatile LONG *)a, x)
#else
#error Unsupported system
#endif

typedef struct shmStruct_st {
  size_t nprocesses;
  int barrier;
  int sense;
  int devices[MAX_DEVICES];
  cudaMemPoolPtrExportData exportPtrData[MAX_DEVICES];
} shmStruct;

__global__ void simpleKernel(char *ptr, int sz, char val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < sz; idx += (gridDim.x * blockDim.x)) {
    ptr[idx] = val;
  }
}

static void barrierWait(volatile int *barrier, volatile int *sense,
                        unsigned int n) {
  int count;

  // Check-in
  count = cpu_atomic_add32(barrier, 1);
  if (count == n)  // Last one in
    *sense = 1;
  while (!*sense)
    ;

  // Check-out
  count = cpu_atomic_add32(barrier, -1);
  if (count == 0)  // Last one out
    *sense = 0;
  while (*sense)
    ;
}

static void childProcess(int id) {
  volatile shmStruct *shm = NULL;
  cudaStream_t stream;
  sharedMemoryInfo info;
  size_t procCount, i;
  int blocks = 0;
  int threads = 128;
  cudaDeviceProp prop;
  std::vector<void *> ptrs;

  std::vector<char> verification_buffer(DATA_SIZE);

  ipcHandle *ipcChildHandle = NULL;
  checkIpcErrors(ipcOpenSocket(ipcChildHandle));

  if (sharedMemoryOpen(shmName, sizeof(shmStruct), &info) != 0) {
    printf("Failed to create shared memory slab\n");
    exit(EXIT_FAILURE);
  }
  shm = (volatile shmStruct *)info.addr;
  procCount = shm->nprocesses;

  barrierWait(&shm->barrier, &shm->sense, (unsigned int)(procCount + 1));

  // Receive all allocation handles shared by Parent.
  std::vector<ShareableHandle> shHandle(shm->nprocesses);
  checkIpcErrors(ipcRecvShareableHandles(ipcChildHandle, shHandle));

  checkCudaErrors(cudaSetDevice(shm->devices[id]));
  checkCudaErrors(cudaGetDeviceProperties(&prop, shm->devices[id]));
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, simpleKernel, threads, 0));
  blocks *= prop.multiProcessorCount;

  std::vector<cudaMemPool_t> pools(shm->nprocesses);

  cudaMemAllocationHandleType handleType = cudaMemHandleTypePosixFileDescriptor;

  // Import mem pools from all the devices created in the master
  // process using shareable handles received via socket
  // and import the pointer to the allocated buffer using
  // exportData filled in shared memory by the master process.
  for (i = 0; i < procCount; i++) {
    checkCudaErrors(cudaMemPoolImportFromShareableHandle(
        &pools[i], (void *)shHandle[i], handleType, 0));

    cudaMemAccessFlags accessFlags;
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    location.id = shm->devices[id];
    checkCudaErrors(cudaMemPoolGetAccess(&accessFlags, pools[i], &location));
    if (accessFlags != cudaMemAccessFlagsProtReadWrite) {
      cudaMemAccessDesc desc;
      memset(&desc, 0, sizeof(cudaMemAccessDesc));
      desc.location.type = cudaMemLocationTypeDevice;
      desc.location.id = shm->devices[id];
      desc.flags = cudaMemAccessFlagsProtReadWrite;
      checkCudaErrors(cudaMemPoolSetAccess(pools[i], &desc, 1));
    }

    // Import the allocation from each memory pool by iterating over exportData
    // until import is success.
    for (int j = 0; j < procCount; j++) {
      void *ptr = NULL;
      // Import the allocation using the opaque export data retrieved through
      // the shared memory".
      cudaError_t ret = cudaMemPoolImportPointer(
          &ptr, pools[i], (cudaMemPoolPtrExportData *)&shm->exportPtrData[j]);

      if (ret == cudaSuccess) {
        // Pointer import is successful hence add it to the ptrs bag.
        ptrs.push_back(ptr);
        break;
      } else {
        // Reset failure error received from cudaMemPoolImportPointer
        // for further try.
        cudaGetLastError();
      }
    }
    // Since we have imported allocations shared by the parent with us, we can
    // close this ShareableHandle.
    checkIpcErrors(ipcCloseShareableHandle(shHandle[i]));
  }

  // Since we have imported allocations shared by the parent with us, we can
  // close the socket.
  checkIpcErrors(ipcCloseSocket(ipcChildHandle));

  // At each iteration of the loop, each sibling process will push work on
  // their respective devices accessing the next peer mapped buffer allocated
  // by the master process (these can come from other sibling processes as
  // well). To coordinate each process' access, we force the stream to wait for
  // the work already accessing this buffer.
  for (i = 0; i < procCount; i++) {
    size_t bufferId = (i + id) % procCount;

    // Push a simple kernel on it
    simpleKernel<<<blocks, threads, 0, stream>>>((char *)ptrs[bufferId],
                                                 DATA_SIZE, id);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Wait for all my sibling processes to push this stage of their work
    // before proceeding to the next. This prevents siblings from racing
    // ahead and clobbering the recorded event or waiting on the wrong
    // recorded event.
    barrierWait(&shm->barrier, &shm->sense, (unsigned int)procCount);
    if (id == 0) {
      printf("Step %lld done\n", (unsigned long long)i);
    }
  }

  // Now wait for my buffer to be ready so I can copy it locally and verify it
  checkCudaErrors(cudaMemcpyAsync(&verification_buffer[0], ptrs[id], DATA_SIZE,
                                  cudaMemcpyDeviceToHost, stream));

  // And wait for all the queued up work to complete
  checkCudaErrors(cudaStreamSynchronize(stream));

  printf("Process %d: verifying...\n", id);

  // The contents should have the id of the sibling just after me
  char compareId = (char)((id + 1) % procCount);
  for (unsigned long long j = 0; j < DATA_SIZE; j++) {
    if (verification_buffer[j] != compareId) {
      printf("Process %d: Verification mismatch at %lld: %d != %d\n", id, j,
             (int)verification_buffer[j], (int)compareId);
    }
  }

  // Clean up!
  for (i = 0; i < procCount; i++) {
    // Free the memory before the exporter process frees it
    checkCudaErrors(cudaFreeAsync(ptrs[i], stream));
  }

  // And wait for all the queued up work to complete
  checkCudaErrors(cudaStreamSynchronize(stream));
  checkCudaErrors(cudaStreamDestroy(stream));

  printf("Process %d complete!\n", id);
}

static void parentProcess(char *app) {
  sharedMemoryInfo info;
  int devCount, i;
  volatile shmStruct *shm = NULL;
  std::vector<void *> ptrs;
  std::vector<Process> processes;

  checkCudaErrors(cudaGetDeviceCount(&devCount));
  std::vector<CUdevice> devices(devCount);
  for (i = 0; i < devCount; i++) {
    cuDeviceGet(&devices[i], i);
  }

  if (sharedMemoryCreate(shmName, sizeof(*shm), &info) != 0) {
    printf("Failed to create shared memory slab\n");
    exit(EXIT_FAILURE);
  }
  shm = (volatile shmStruct *)info.addr;
  memset((void *)shm, 0, sizeof(*shm));

  // Pick all the devices that can access each other's memory for this test
  // Keep in mind that CUDA has minimal support for fork() without a
  // corresponding exec() in the child process, but in this case our
  // spawnProcess will always exec, so no need to worry.
  for (i = 0; i < devCount; i++) {
    bool allPeers = true;
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, i));

    int isMemPoolSupported = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&isMemPoolSupported,
                                           cudaDevAttrMemoryPoolsSupported, i));
    // CUDA IPC is only supported on devices with unified addressing
    if (!isMemPoolSupported) {
      printf("Device %d does not support cuda memory pools, skipping...\n", i);
      continue;
    }
    int deviceSupportsIpcHandle = 0;
#if defined(__linux__)
    checkCudaErrors(cuDeviceGetAttribute(
        &deviceSupportsIpcHandle,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
        devices[i]));
#else
    cuDeviceGetAttribute(&deviceSupportsIpcHandle,
                         CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED,
                         devices[i]);
#endif

    if (!deviceSupportsIpcHandle) {
      printf("Device %d does not support CUDA IPC Handle, skipping...\n", i);
      continue;
    }
    // This sample requires two processes accessing each device, so we need
    // to ensure exclusive or prohibited mode is not set
    if (prop.computeMode != cudaComputeModeDefault) {
      printf("Device %d is in an unsupported compute mode for this sample\n",
             i);
      continue;
    }
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    // CUDA IPC on Windows is only supported on TCC
    if (!prop.tccDriver) {
      printf("Device %d is not in TCC mode\n", i);
      continue;
    }
#endif

    for (int j = 0; j < shm->nprocesses; j++) {
      int canAccessPeerIJ, canAccessPeerJI;
      checkCudaErrors(
          cudaDeviceCanAccessPeer(&canAccessPeerJI, shm->devices[j], i));
      checkCudaErrors(
          cudaDeviceCanAccessPeer(&canAccessPeerIJ, i, shm->devices[j]));
      if (!canAccessPeerIJ || !canAccessPeerJI) {
        allPeers = false;
        break;
      }
    }
    if (allPeers) {
      // Enable peers here.  This isn't necessary for IPC, but it will
      // setup the peers for the device.  For systems that only allow 8
      // peers per GPU at a time, this acts to remove devices from CanAccessPeer
      for (int j = 0; j < shm->nprocesses; j++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaDeviceEnablePeerAccess(shm->devices[j], 0));
        checkCudaErrors(cudaSetDevice(shm->devices[j]));
        checkCudaErrors(cudaDeviceEnablePeerAccess(i, 0));
      }
      shm->devices[shm->nprocesses++] = i;
      if (shm->nprocesses >= MAX_DEVICES) break;
    } else {
      printf(
          "Device %d is not peer capable with some other selected peers, "
          "skipping\n",
          i);
    }
  }

  if (shm->nprocesses == 0) {
    printf("No CUDA devices support IPC\n");
    exit(EXIT_WAIVED);
  }

  std::vector<ShareableHandle> shareableHandles(shm->nprocesses);
  std::vector<cudaStream_t> streams(shm->nprocesses);
  std::vector<cudaMemPool_t> pools(shm->nprocesses);

  // Now allocate memory for each process and fill the shared
  // memory buffer with the export data and get memPool handles to communicate
  for (i = 0; i < shm->nprocesses; i++) {
    void *ptr = NULL;
    checkCudaErrors(cudaSetDevice(shm->devices[i]));
    checkCudaErrors(
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    // Allocate an explicit pool with IPC capabilities
    cudaMemPoolProps poolProps;
    memset(&poolProps, 0, sizeof(cudaMemPoolProps));
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.handleTypes = cudaMemHandleTypePosixFileDescriptor;

    poolProps.location.type = cudaMemLocationTypeDevice;
    poolProps.location.id = shm->devices[i];

    checkCudaErrors(cudaMemPoolCreate(&pools[i], &poolProps));

    // Query the shareable handle for the pool
    cudaMemAllocationHandleType handleType =
        cudaMemHandleTypePosixFileDescriptor;
    // Allocate memory in a stream from the pool just created
    checkCudaErrors(cudaMallocAsync(&ptr, DATA_SIZE, pools[i], streams[i]));

    checkCudaErrors(cudaMemPoolExportToShareableHandle(
        &shareableHandles[i], pools[i], handleType, 0));

    // Get the opaque ‘bag-of-bits’ representing the allocation
    memset((void *)&shm->exportPtrData[i], 0, sizeof(cudaMemPoolPtrExportData));
    checkCudaErrors(cudaMemPoolExportPointer(
        (cudaMemPoolPtrExportData *)&shm->exportPtrData[i], ptr));
    ptrs.push_back(ptr);
  }

  // Launch the child processes!
  for (i = 0; i < shm->nprocesses; i++) {
    char devIdx[10];
    char *const args[] = {app, devIdx, NULL};
    Process process;

    SPRINTF(devIdx, "%d", i);

    if (spawnProcess(&process, app, args)) {
      printf("Failed to create process\n");
      exit(EXIT_FAILURE);
    }

    processes.push_back(process);
  }

  barrierWait(&shm->barrier, &shm->sense, (unsigned int)(shm->nprocesses + 1));

  ipcHandle *ipcParentHandle = NULL;
  checkIpcErrors(ipcCreateSocket(ipcParentHandle, ipcName, processes));
  checkIpcErrors(
      ipcSendShareableHandles(ipcParentHandle, shareableHandles, processes));

  // Close the shareable handles as they are not needed anymore.
  for (int i = 0; i < shm->nprocesses; i++) {
    checkIpcErrors(ipcCloseShareableHandle(shareableHandles[i]));
  }
  checkIpcErrors(ipcCloseSocket(ipcParentHandle));

  // And wait for them to finish
  for (i = 0; i < processes.size(); i++) {
    if (waitProcess(&processes[i]) != EXIT_SUCCESS) {
      printf("Process %d failed!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  // Clean up!
  for (i = 0; i < shm->nprocesses; i++) {
    checkCudaErrors(cudaSetDevice(shm->devices[i]));
    checkCudaErrors(cudaFreeAsync(ptrs[i], streams[i]));
    checkCudaErrors(cudaStreamSynchronize(streams[i]));
    checkCudaErrors(cudaMemPoolDestroy(pools[i]));
  }

  sharedMemoryClose(&info);
}

// Host code
int main(int argc, char **argv) {
#if defined(__arm__) || defined(__aarch64__) || defined(WIN32) || \
    defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  printf("Not supported on ARM\n");
  return EXIT_WAIVED;
#else
  if (argc == 1) {
    parentProcess(argv[0]);
  } else {
    childProcess(atoi(argv[1]));
  }
  return EXIT_SUCCESS;
#endif
}
