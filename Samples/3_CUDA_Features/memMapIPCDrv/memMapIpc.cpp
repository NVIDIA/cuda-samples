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
 * using cuMemMap APIs and with one process per GPU for computation.
 */

#include <stdio.h>
#include <string.h>
#include <cstring>
#include <iostream>
#include "cuda.h"

#include "helper_multiprocess.h"

// includes, project
#include <helper_functions.h>
#include "helper_cuda_drvapi.h"

// includes, CUDA
#include <builtin_types.h>

using namespace std;

// For direct NVLINK and PCI-E peers, at max 8 simultaneous peers are allowed
// For NVSWITCH connected peers like DGX-2, simultaneous peers are not limited
// in the same way.
#define MAX_DEVICES (32)

#define PROCESSES_PER_DEVICE 1
#define DATA_BUF_SIZE 4ULL * 1024ULL * 1024ULL

static const char ipcName[] = "memmap_ipc_pipe";
static const char shmName[] = "memmap_ipc_shm";

typedef struct shmStruct_st {
  size_t nprocesses;
  int barrier;
  int sense;
} shmStruct;

bool findModulePath(const char *, string &, char **, string &);

// define input ptx file for different platforms
#if defined(_WIN64) || defined(__LP64__)
#define PTX_FILE "memMapIpc_kernel64.ptx"
#else
#define PTX_FILE "memMapIpc_kernel32.ptx"
#endif

// `ipcHandleTypeFlag` specifies the platform specific handle type this sample
// uses for importing and exporting memory allocation. On Linux this sample
// specifies the type as CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR meaning that
// file descriptors will be used. On Windows this sample specifies the type as
// CU_MEM_HANDLE_TYPE_WIN32 meaning that NT HANDLEs will be used. The
// ipcHandleTypeFlag variable is a convenience variable and is passed by value
// to individual requests.
#if defined(__linux__)
CUmemAllocationHandleType ipcHandleTypeFlag =
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
CUmemAllocationHandleType ipcHandleTypeFlag = CU_MEM_HANDLE_TYPE_WIN32;
#endif

#if defined(__linux__)
#define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define cpu_atomic_add32(a, x) InterlockedAdd((volatile LONG *)a, x)
#else
#error Unsupported system
#endif

CUmodule cuModule;
CUfunction _memMapIpc_kernel;

static void barrierWait(volatile int *barrier, volatile int *sense,
                        unsigned int n) {
  int count;

  // Check-in
  count = cpu_atomic_add32(barrier, 1);
  if (count == n) {  // Last one in
    *sense = 1;
  }
  while (!*sense)
    ;

  // Check-out
  count = cpu_atomic_add32(barrier, -1);
  if (count == 0) {  // Last one out
    *sense = 0;
  }
  while (*sense)
    ;
}

// Windows-specific LPSECURITYATTRIBUTES
void getDefaultSecurityDescriptor(CUmemAllocationProp *prop) {
#if defined(__linux__)
  return;
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  static const char sddl[] = "D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)";
  static OBJECT_ATTRIBUTES objAttributes;
  static bool objAttributesConfigured = false;

  if (!objAttributesConfigured) {
    PSECURITY_DESCRIPTOR secDesc;
    BOOL result = ConvertStringSecurityDescriptorToSecurityDescriptorA(
        sddl, SDDL_REVISION_1, &secDesc, NULL);
    if (result == 0) {
      printf("IPC failure: getDefaultSecurityDescriptor Failed! (%d)\n",
             GetLastError());
    }

    InitializeObjectAttributes(&objAttributes, NULL, 0, NULL, secDesc);

    objAttributesConfigured = true;
  }

  prop->win32HandleMetaData = &objAttributes;
  return;
#endif
}

static void memMapAllocateAndExportMemory(
    unsigned char backingDevice, size_t allocSize,
    std::vector<CUmemGenericAllocationHandle> &allocationHandles,
    std::vector<ShareableHandle> &shareableHandles) {
  // This property structure describes the physical location where the memory
  // will be allocated via cuMemCreate along with additional properties.
  CUmemAllocationProp prop = {};

  // The allocations will be device pinned memory backed on backingDevice and
  // exportable with the specified handle type.
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

  // Back all allocations on backingDevice.
  prop.location.id = (int)backingDevice;

  // Passing a requestedHandleTypes indicates intention to export this
  // allocation to a platform-specific handle. This sample requests a file
  // descriptor on Linux and NT Handle on Windows.
  prop.requestedHandleTypes = ipcHandleTypeFlag;

  // Get the minimum granularity supported for allocation with cuMemCreate()
  size_t granularity = 0;
  checkCudaErrors(cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  if (allocSize % granularity) {
    printf(
        "Allocation size is not a multiple of minimum supported granularity "
        "for this device. Exiting...\n");
    exit(EXIT_FAILURE);
  }

  // Windows-specific LPSECURITYATTRIBUTES is required when
  // CU_MEM_HANDLE_TYPE_WIN32 is used. The security attribute defines the scope
  // of which exported allocations may be tranferred to other processes. For all
  // other handle types, pass NULL.
  getDefaultSecurityDescriptor(&prop);

  for (int i = 0; i < allocationHandles.size(); i++) {
    // Create the allocation as a pinned allocation on device specified in
    // prop.location.id
    checkCudaErrors(cuMemCreate(&allocationHandles[i], allocSize, &prop, 0));

    // Export the allocation to a platform-specific handle. The type of handle
    // requested here must match the requestedHandleTypes field in the prop
    // structure passed to cuMemCreate.
    checkCudaErrors(cuMemExportToShareableHandle((void *)&shareableHandles[i],
                                                 allocationHandles[i],
                                                 ipcHandleTypeFlag, 0));
  }
}

static void memMapImportAndMapMemory(
    CUdeviceptr d_ptr, size_t mapSize,
    std::vector<ShareableHandle> &shareableHandles, int mapDevice) {
  std::vector<CUmemGenericAllocationHandle> allocationHandles;
  allocationHandles.resize(shareableHandles.size());

  // The accessDescriptor will describe the mapping requirement for the
  // mapDevice passed as argument
  CUmemAccessDesc accessDescriptor;

  // Specify location for mapping the imported allocations.
  accessDescriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDescriptor.location.id = mapDevice;

  // Specify both read and write accesses.
  accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  for (int i = 0; i < shareableHandles.size(); i++) {
    // Import the memory allocation back into a CUDA handle from the platform
    // specific handle.
    checkCudaErrors(cuMemImportFromShareableHandle(
        &allocationHandles[i], (void *)(uintptr_t)shareableHandles[i],
        ipcHandleTypeFlag));

    // Assign the chunk to the appropriate VA range and release the handle.
    // After mapping the memory, it can be referenced by virtual address.
    checkCudaErrors(
        cuMemMap(d_ptr + (i * mapSize), mapSize, 0, allocationHandles[i], 0));

    // Since we do not need to make any other mappings of this memory or export
    // it, we no longer need and can release the allocationHandle. The
    // allocation will be kept live until it is unmapped.
    checkCudaErrors(cuMemRelease(allocationHandles[i]));
  }

  // Retain peer access and map all chunks to mapDevice
  checkCudaErrors(cuMemSetAccess(d_ptr, shareableHandles.size() * mapSize,
                                 &accessDescriptor, 1));
}

static void memMapUnmapAndFreeMemory(CUdeviceptr dptr, size_t size) {
  CUresult status = CUDA_SUCCESS;

  // Unmap the mapped virtual memory region
  // Since the handles to the mapped backing stores have already been released
  // by cuMemRelease, and these are the only/last mappings referencing them,
  // The backing stores will be freed.
  // Since the memory has been unmapped after this call, accessing the specified
  // va range will result in a fault (unitll it is remapped).
  checkCudaErrors(cuMemUnmap(dptr, size));

  // Free the virtual address region.  This allows the virtual address region
  // to be reused by future cuMemAddressReserve calls.  This also allows the
  // virtual address region to be used by other allocation made through
  // opperating system calls like malloc & mmap.
  checkCudaErrors(cuMemAddressFree(dptr, size));
}

static void memMapGetDeviceFunction(char **argv) {
  // first search for the module path before we load the results
  string module_path, ptx_source;
  if (!findModulePath(PTX_FILE, module_path, argv, ptx_source)) {
    if (!findModulePath("memMapIpc_kernel.cubin", module_path, argv,
                        ptx_source)) {
      printf(
          "> findModulePath could not find <simpleMemMapIpc> ptx or cubin\n");
      exit(EXIT_FAILURE);
    }
  } else {
    printf("> initCUDA loading module: <%s>\n", module_path.c_str());
  }

  // Create module from binary file (PTX or CUBIN)
  if (module_path.rfind("ptx") != string::npos) {
    // in this branch we use compilation with parameters
    const unsigned int jitNumOptions = 3;
    CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
    void **jitOptVals = new void *[jitNumOptions];
    // set up size of compilation log buffer
    jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    int jitLogBufferSize = 1024;
    jitOptVals[0] = (void *)(size_t)jitLogBufferSize;
    // set up pointer to the compilation log buffer
    jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
    char *jitLogBuffer = new char[jitLogBufferSize];
    jitOptVals[1] = jitLogBuffer;
    // set up pointer to set the Maximum # of registers for a particular kernel
    jitOptions[2] = CU_JIT_MAX_REGISTERS;
    int jitRegCount = 32;
    jitOptVals[2] = (void *)(size_t)jitRegCount;
    checkCudaErrors(cuModuleLoadDataEx(&cuModule, ptx_source.c_str(),
                                       jitNumOptions, jitOptions,
                                       (void **)jitOptVals));
    printf("> PTX JIT log:\n%s\n", jitLogBuffer);
  } else {
    checkCudaErrors(cuModuleLoad(&cuModule, module_path.c_str()));
  }

  // Get function handle from module
  checkCudaErrors(
      cuModuleGetFunction(&_memMapIpc_kernel, cuModule, "memMapIpc_kernel"));
}

static void childProcess(int devId, int id, char **argv) {
  volatile shmStruct *shm = NULL;
  sharedMemoryInfo info;
  ipcHandle *ipcChildHandle = NULL;
  int blocks = 0;
  int threads = 128;

  checkIpcErrors(ipcOpenSocket(ipcChildHandle));

  if (sharedMemoryOpen(shmName, sizeof(shmStruct), &info) != 0) {
    printf("Failed to create shared memory slab\n");
    exit(EXIT_FAILURE);
  }
  shm = (volatile shmStruct *)info.addr;
  int procCount = (int)shm->nprocesses;

  barrierWait(&shm->barrier, &shm->sense, (unsigned int)(procCount + 1));

  // Receive all allocation handles shared by Parent.
  std::vector<ShareableHandle> shHandle(procCount);
  checkIpcErrors(ipcRecvShareableHandles(ipcChildHandle, shHandle));

  CUcontext ctx;
  CUdevice device;
  CUstream stream;
  int multiProcessorCount;

  checkCudaErrors(cuDeviceGet(&device, devId));
  checkCudaErrors(cuCtxCreate(&ctx, 0, device));
  checkCudaErrors(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  // Obtain kernel function for the sample
  memMapGetDeviceFunction(argv);

  checkCudaErrors(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, _memMapIpc_kernel, threads, 0));
  checkCudaErrors(cuDeviceGetAttribute(
      &multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  blocks *= multiProcessorCount;

  CUdeviceptr d_ptr = 0ULL;

  // Reserve the required contiguous VA space for the allocations
  checkCudaErrors(cuMemAddressReserve(&d_ptr, procCount * DATA_BUF_SIZE,
                                      DATA_BUF_SIZE, 0, 0));

  // Import the memory allocations shared by the parent with us and map them in
  // our address space.
  memMapImportAndMapMemory(d_ptr, DATA_BUF_SIZE, shHandle, devId);

  // Since we have imported allocations shared by the parent with us, we can
  // close all the ShareableHandles.
  for (int i = 0; i < procCount; i++) {
    checkIpcErrors(ipcCloseShareableHandle(shHandle[i]));
  }
  checkIpcErrors(ipcCloseSocket(ipcChildHandle));

  for (int i = 0; i < procCount; i++) {
    size_t bufferId = (i + id) % procCount;

    // Build arguments to be passed to cuda kernel.
    CUdeviceptr ptr = d_ptr + (bufferId * DATA_BUF_SIZE);
    int size = DATA_BUF_SIZE;
    char val = (char)id;

    void *args[] = {&ptr, &size, &val};

    // Push a simple kernel on th buffer.
    checkCudaErrors(cuLaunchKernel(_memMapIpc_kernel, blocks, 1, 1, threads, 1,
                                   1, 0, stream, args, 0));
    checkCudaErrors(cuStreamSynchronize(stream));

    // Wait for all my sibling processes to push this stage of their work
    // before proceeding to the next. This makes the data in the buffer
    // deterministic.
    barrierWait(&shm->barrier, &shm->sense, (unsigned int)procCount);
    if (id == 0) {
      printf("Step %lld done\n", (unsigned long long)i);
    }
  }

  printf("Process %d: verifying...\n", id);

  // Copy the data onto host and verify value if it matches expected value or
  // not.
  std::vector<char> verification_buffer(DATA_BUF_SIZE);
  checkCudaErrors(cuMemcpyDtoHAsync(&verification_buffer[0],
                                    d_ptr + (id * DATA_BUF_SIZE), DATA_BUF_SIZE,
                                    stream));
  checkCudaErrors(cuStreamSynchronize(stream));

  // The contents should have the id of the sibling just after me
  char compareId = (char)((id + 1) % procCount);
  for (unsigned long long j = 0; j < DATA_BUF_SIZE; j++) {
    if (verification_buffer[j] != compareId) {
      printf("Process %d: Verification mismatch at %lld: %d != %d\n", id, j,
             (int)verification_buffer[j], (int)compareId);
      break;
    }
  }

  // Clean up!
  checkCudaErrors(cuStreamDestroy(stream));
  checkCudaErrors(cuCtxDestroy(ctx));

  // Unmap the allocations from our address space. Unmapping will also free the
  // handle as we have already released the imported handle with the call to
  // cuMemRelease. Finally, free up the Virtual Address space we reserved with
  // cuMemAddressReserve.
  memMapUnmapAndFreeMemory(d_ptr, procCount * DATA_BUF_SIZE);

  exit(EXIT_SUCCESS);
}

static void parentProcess(char *app) {
  int devCount, i, nprocesses = 0;
  volatile shmStruct *shm = NULL;
  sharedMemoryInfo info;
  std::vector<Process> processes;

  checkCudaErrors(cuDeviceGetCount(&devCount));
  std::vector<CUdevice> devices(devCount);

  if (sharedMemoryCreate(shmName, sizeof(*shm), &info) != 0) {
    printf("Failed to create shared memory slab\n");
    exit(EXIT_FAILURE);
  }

  shm = (volatile shmStruct *)info.addr;
  memset((void *)shm, 0, sizeof(*shm));

  for (i = 0; i < devCount; i++) {
    checkCudaErrors(cuDeviceGet(&devices[i], i));
  }

  std::vector<CUcontext> ctxs;
  std::vector<unsigned char> selectedDevices;

  // Pick all the devices that can access each other's memory for this test
  // Keep in mind that CUDA has minimal support for fork() without a
  // corresponding exec() in the child process, but in this case our
  // spawnProcess will always exec, so no need to worry.
  for (i = 0; i < devCount; i++) {
    bool allPeers = true;
    int deviceComputeMode;
    int deviceSupportsIpcHandle;
    int attributeVal = 0;

    checkCudaErrors(cuDeviceGet(&devices[i], i));
    checkCudaErrors(cuDeviceGetAttribute(
        &deviceComputeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, devices[i]));
    checkCudaErrors(cuDeviceGetAttribute(
        &attributeVal, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
        devices[i]));
#if defined(__linux__)
    checkCudaErrors(cuDeviceGetAttribute(
        &deviceSupportsIpcHandle,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
        devices[i]));
#else
    checkCudaErrors(cuDeviceGetAttribute(
        &deviceSupportsIpcHandle,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, devices[i]));
#endif
    // Check that the selected device supports virtual address management
    if (attributeVal == 0) {
      printf("Device %d doesn't support VIRTUAL ADDRESS MANAGEMENT.\n",
             devices[i]);
      continue;
    }

    // This sample requires two processes accessing each device, so we need
    // to ensure exclusive or prohibited mode is not set
    if (deviceComputeMode != CU_COMPUTEMODE_DEFAULT) {
      printf("Device %d is in an unsupported compute mode for this sample\n",
             i);
      continue;
    }

    if (!deviceSupportsIpcHandle) {
      printf(
          "Device %d does not support requested handle type for IPC, "
          "skipping...\n",
          i);
      continue;
    }

    for (int j = 0; j < nprocesses; j++) {
      int canAccessPeerIJ, canAccessPeerJI;
      checkCudaErrors(
          cuDeviceCanAccessPeer(&canAccessPeerJI, devices[j], devices[i]));
      checkCudaErrors(
          cuDeviceCanAccessPeer(&canAccessPeerIJ, devices[i], devices[j]));
      if (!canAccessPeerIJ || !canAccessPeerJI) {
        allPeers = false;
        break;
      }
    }
    if (allPeers) {
      CUcontext ctx;
      checkCudaErrors(cuCtxCreate(&ctx, 0, devices[i]));
      ctxs.push_back(ctx);

      // Enable peers here.  This isn't necessary for IPC, but it will
      // setup the peers for the device.  For systems that only allow 8
      // peers per GPU at a time, this acts to remove devices from CanAccessPeer
      for (int j = 0; j < nprocesses; j++) {
        checkCudaErrors(cuCtxSetCurrent(ctxs[i]));
        checkCudaErrors(cuCtxEnablePeerAccess(ctxs[j], 0));
        checkCudaErrors(cuCtxSetCurrent(ctxs[j]));
        checkCudaErrors(cuCtxEnablePeerAccess(ctxs[i], 0));
      }
      selectedDevices.push_back(i);
      nprocesses++;
      if (nprocesses >= MAX_DEVICES) {
        break;
      }
    } else {
      printf(
          "Device %d is not peer capable with some other selected peers, "
          "skipping\n",
          i);
    }
  }

  for (int i = 0; i < ctxs.size(); ++i) {
    checkCudaErrors(cuCtxDestroy(ctxs[i]));
  };

  if (nprocesses == 0) {
    printf("No CUDA devices support IPC\n");
    exit(EXIT_WAIVED);
  }
  shm->nprocesses = nprocesses;

  unsigned char firstSelectedDevice = selectedDevices[0];

  std::vector<ShareableHandle> shHandles(nprocesses);
  std::vector<CUmemGenericAllocationHandle> allocationHandles(nprocesses);

  // Allocate `nprocesses` number of memory chunks and obtain a shareable handle
  // for each allocation. Share all memory allocations with all children.
  memMapAllocateAndExportMemory(firstSelectedDevice, DATA_BUF_SIZE,
                                allocationHandles, shHandles);

  // Launch the child processes!
  for (i = 0; i < nprocesses; i++) {
    char devIdx[10];
    char procIdx[10];
    char *const args[] = {app, devIdx, procIdx, NULL};
    Process process;

    SPRINTF(devIdx, "%d", selectedDevices[i]);
    SPRINTF(procIdx, "%d", i);

    if (spawnProcess(&process, app, args)) {
      printf("Failed to create process\n");
      exit(EXIT_FAILURE);
    }

    processes.push_back(process);
  }

  barrierWait(&shm->barrier, &shm->sense, (unsigned int)(nprocesses + 1));

  ipcHandle *ipcParentHandle = NULL;
  checkIpcErrors(ipcCreateSocket(ipcParentHandle, ipcName, processes));
  checkIpcErrors(
      ipcSendShareableHandles(ipcParentHandle, shHandles, processes));

  // Close the shareable handles as they are not needed anymore.
  for (int i = 0; i < nprocesses; i++) {
    checkIpcErrors(ipcCloseShareableHandle(shHandles[i]));
  }

  // And wait for them to finish
  for (i = 0; i < processes.size(); i++) {
    if (waitProcess(&processes[i]) != EXIT_SUCCESS) {
      printf("Process %d failed!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  for (i = 0; i < nprocesses; i++) {
    checkCudaErrors(cuMemRelease(allocationHandles[i]));
  }

  checkIpcErrors(ipcCloseSocket(ipcParentHandle));
  sharedMemoryClose(&info);
}

// Host code
int main(int argc, char **argv) {
#if defined(__arm__) || defined(__aarch64__)
  printf("Not supported on ARM\n");
  return EXIT_WAIVED;
#else
  // Initialize
  checkCudaErrors(cuInit(0));

  if (argc == 1) {
    parentProcess(argv[0]);
  } else {
    childProcess(atoi(argv[1]), atoi(argv[2]), argv);
  }
  return EXIT_SUCCESS;
#endif
}

bool inline findModulePath(const char *module_file, string &module_path,
                           char **argv, string &ptx_source) {
  char *actual_path = sdkFindFilePath(module_file, argv[0]);

  if (actual_path) {
    module_path = actual_path;
  } else {
    printf("> findModulePath file not found: <%s> \n", module_file);
    return false;
  }

  if (module_path.empty()) {
    printf("> findModulePath could not find file: <%s> \n", module_file);
    return false;
  } else {
    printf("> findModulePath found file at <%s>\n", module_path.c_str());

    if (module_path.rfind(".ptx") != string::npos) {
      FILE *fp = fopen(module_path.c_str(), "rb");
      fseek(fp, 0, SEEK_END);
      int file_size = ftell(fp);
      char *buf = new char[file_size + 1];
      fseek(fp, 0, SEEK_SET);
      fread(buf, sizeof(char), file_size, fp);
      fclose(fp);
      buf[file_size] = '\0';
      ptx_source = buf;
      delete[] buf;
    }

    return true;
  }
}