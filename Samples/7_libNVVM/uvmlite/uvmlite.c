// Copyright (c) 2014-2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <assert.h>
#include <builtin_types.h>
#include <cuda.h>
#include <math.h>
#include <nvvm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define ERROR_IF(expr)                                              \
  if (expr) {                                                       \
    fprintf(stderr, "Failed check at %s:%d\n", __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                             \
  }

// If 'err' is non-zero, emit an error message and exit.
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
static void __checkCudaErrors(CUresult err, const char *filename, int line) {
  assert(filename);
  if (CUDA_SUCCESS != err) {
    const char *ename = NULL;
    const CUresult res = cuGetErrorName(err, &ename);
    fprintf(stderr,
            "CUDA API Error %04d: \"%s\" from file <%s>, "
            "line %i.\n",
            err, ((CUDA_SUCCESS == res) ? ename : "Unknown"), filename, line);
    exit(err);
  }
}

// Compile the NVVM IR into PTX.
static char *generatePTX(const char *ll, size_t size, const char *filename,
                         int devMajor, int devMinor) {
  assert(ll && filename);

  // Create a program instance for libNVVM.
  nvvmProgram program;
  nvvmResult result = nvvmCreateProgram(&program);
  if (result != NVVM_SUCCESS) {
    fprintf(stderr, "nvvmCreateProgram: Failed\n");
    exit(EXIT_FAILURE);
  }

  // Add the NVVM IR as a module to our libNVVM program.
  result = nvvmAddModuleToProgram(program, ll, size, filename);
  if (result != NVVM_SUCCESS) {
    fprintf(stderr, "nvvmAddModuleToProgram: Failed\n");
    exit(EXIT_FAILURE);
  }

  // Dynamically construct the compute capability.
  char arch[32] = {0};
  snprintf(arch, sizeof(arch) - 1, "-arch=compute_%d%d", devMajor, devMinor);

  // Compile the IR into PTX.
  const char *options[] = {arch};
  result = nvvmCompileProgram(program, 1, options);
  if (result != NVVM_SUCCESS) {
    char *Msg = NULL;
    size_t LogSize;
    fprintf(stderr, "nvvmCompileProgram: Failed\n");
    nvvmGetProgramLogSize(program, &LogSize);
    Msg = (char *)malloc(LogSize);
    nvvmGetProgramLog(program, Msg);
    fprintf(stderr, "%s\n", Msg);
    free(Msg);
    exit(EXIT_FAILURE);
  }

  size_t ptxSize = 0;
  result = nvvmGetCompiledResultSize(program, &ptxSize);
  if (result != NVVM_SUCCESS) {
    fprintf(stderr, "nvvmGetCompiledResultSize: Failed\n");
    exit(EXIT_FAILURE);
  }

  char *ptx = malloc(ptxSize);
  assert(ptx);
  result = nvvmGetCompiledResult(program, ptx);
  if (result != NVVM_SUCCESS) {
    fprintf(stderr, "nvvmGetCompiledResult: Failed\n");
    free(ptx);
    exit(EXIT_FAILURE);
  }

  result = nvvmDestroyProgram(&program);
  if (result != NVVM_SUCCESS) {
    fprintf(stderr, "nvvmDestroyProgram: Failed\n");
    free(ptx);
    exit(EXIT_FAILURE);
  }

  return ptx;
}

static char *loadProgramSource(const char *filename, size_t *size) {
  assert(filename && size);
  *size = 0;
  char *source = NULL;
  FILE *fh = fopen(filename, "rb");
  if (fh) {
    struct stat statbuf;
    stat(filename, &statbuf);
    source = (char *)malloc(statbuf.st_size + 1);
    if (source) {
      fread(source, statbuf.st_size, 1, fh);
      source[statbuf.st_size] = 0;
      *size = statbuf.st_size + 1;
    }
  } else {
    fprintf(stderr, "Error reading file %s\n", filename);
    exit(EXIT_FAILURE);
  }
  return source;
}

// Return the device compute capability in major and minor.
static CUdevice cudaDeviceInit(int *major, int *minor) {
  assert(major && minor);
  // Count the number of CUDA compute capable devices..
  CUresult err = cuInit(0);
  int deviceCount = 0;
  if (CUDA_SUCCESS == err)
    checkCudaErrors(cuDeviceGetCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
    exit(EXIT_FAILURE);
  }

  // Get the first device discovered (device 0) and print its name.
  CUdevice cuDevice = 0;
  checkCudaErrors(cuDeviceGet(&cuDevice, 0));
  char name[128] = {0};
  checkCudaErrors(cuDeviceGetName(name, sizeof(name), cuDevice));
  printf("Using CUDA Device [0]: %s\n", name);

  // Get and test the compute capability.
  checkCudaErrors(cuDeviceGetAttribute(
      major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  checkCudaErrors(cuDeviceGetAttribute(
      minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
  printf("compute capability = %d.%d\n", *major, *minor);
  if (*major < 5) {
    fprintf(stderr, "Device 0 is not sm_50 or later\n");
    exit(EXIT_FAILURE);
  }

  // Check if managed memory is supported.
  int supportsUvm = 0;
  checkCudaErrors(cuDeviceGetAttribute(
      &supportsUvm, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, cuDevice));
  if (!supportsUvm) {
    printf("This device does not support managed memory.");
    exit(EXIT_SUCCESS);
  }

  // Check if unified addressing is supported (host and device share same
  // the address space).
  int supportsUva = 0;
  checkCudaErrors(cuDeviceGetAttribute(
      &supportsUva, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cuDevice));
  if (!supportsUva) {
    printf("This device does not support a unified address space.");
    exit(EXIT_SUCCESS);
  }

  return cuDevice;
}

static CUresult buildKernel(CUcontext *phContext, CUdevice *phDevice,
                            CUmodule *phModule, CUfunction *phKernel) {
  assert(phContext && phDevice && phModule && phKernel);

  // Initialize CUDA and obtain the device's compute capability.
  int major = 0, minor = 0;
  *phDevice = cudaDeviceInit(&major, &minor);

  // Create a context on the device.
  checkCudaErrors(cuCtxCreate(phContext, 0, *phDevice));

  // Get the NVVM IR from file.
  size_t size = 0;
  const char *filename = "uvmlite64.ll";
  char *ll = loadProgramSource(filename, &size);
  fprintf(stdout, "NVVM IR ll file loaded\n");

  // Use libNVVM to generate PTX.
  char *ptx = generatePTX(ll, size, filename, major, minor);
  fprintf(stdout, "PTX generated:\n");
  fprintf(stdout, "%s\n", ptx);

  // Load module from PTX.
  checkCudaErrors(cuModuleLoadDataEx(phModule, ptx, 0, NULL, NULL));

  // Locate the kernel entry point.
  checkCudaErrors(cuModuleGetFunction(phKernel, *phModule, "test_kernel"));

  free(ll);
  free(ptx);
  return CUDA_SUCCESS;
}

int main(void) {
  const unsigned int nThreads = 1;
  const unsigned int nBlocks = 1;

  // Pointers to the variables in the managed memory.
  // See uvmlite64.ll for their definition.
  CUdeviceptr devp_xxx, devp_yyy;
  size_t size_xxx, size_yyy;
  int *p_xxx, *p_yyy;

  // Initialize the device and get a handle to the kernel
  CUcontext hContext = 0;
  CUdevice hDevice = 0;
  CUmodule hModule = 0;
  CUfunction hKernel = 0;
  checkCudaErrors(buildKernel(&hContext, &hDevice, &hModule, &hKernel));

  // Whether or not a device supports unified addressing may be queried by
  // calling cuDeviceGetAttribute() with the deivce attribute
  // CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING.
  {
    int attrVal;
    checkCudaErrors(cuDeviceGetAttribute(
        &attrVal, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, hDevice));
    ERROR_IF(attrVal != 1);
  }

  // Get the address of the variable xxx, yyy in the managed memory.
  checkCudaErrors(cuModuleGetGlobal(&devp_xxx, &size_xxx, hModule, "xxx"));
  checkCudaErrors(cuModuleGetGlobal(&devp_yyy, &size_yyy, hModule, "yyy"));

  // Whether or not the pointer points to managed memory may be queried by
  // calling cuPointerGetAttribute() with the pointer attribute
  // CU_POINTER_ATTRIBUTE_IS_MANAGED.
  {
    unsigned int attrVal;

    checkCudaErrors(cuPointerGetAttribute(
        &attrVal, CU_POINTER_ATTRIBUTE_IS_MANAGED, devp_xxx));
    ERROR_IF(attrVal != 1);
    checkCudaErrors(cuPointerGetAttribute(
        &attrVal, CU_POINTER_ATTRIBUTE_IS_MANAGED, devp_yyy));
    ERROR_IF(attrVal != 1);
  }

  // Since CUdeviceptr is opaque, it is safe to use cuPointerGetAttribute to get
  // the host pointers.
  {
    void *host_ptr_xxx, *host_ptr_yyy;

    checkCudaErrors(cuPointerGetAttribute(
        &host_ptr_xxx, CU_POINTER_ATTRIBUTE_HOST_POINTER, devp_xxx));
    checkCudaErrors(cuPointerGetAttribute(
        &host_ptr_yyy, CU_POINTER_ATTRIBUTE_HOST_POINTER, devp_yyy));

    p_xxx = (int *)host_ptr_xxx;
    p_yyy = (int *)host_ptr_yyy;
  }

  printf("The initial value of xxx initialized by the device = %d\n", *p_xxx);
  printf("The initial value of yyy initialized by the device = %d\n", *p_yyy);

  ERROR_IF(*p_xxx != 10);
  ERROR_IF(*p_yyy != 100);

  // The host adds 1 and 11 to xxx and yyy.
  *p_xxx += 1;
  *p_yyy += 11;

  printf("The host added 1 and 11 to xxx and yyy.\n");

  // Launch the kernel with the following parameters.
  {
    void *params[] = {(void *)&devp_xxx};
    checkCudaErrors(cuLaunchKernel(hKernel, nBlocks, 1, 1, nThreads, 1, 1, 0,
                                   NULL, params, NULL));
  }
  checkCudaErrors(cuCtxSynchronize());

  printf("kernel added 20 and 30 to xxx and yyy, respectively.\n");
  printf("The final value checked in the host: xxx = %d, yyy = %d\n", *p_xxx,
         *p_yyy);

  if (hModule) {
    checkCudaErrors(cuModuleUnload(hModule));
    hModule = 0;
  }
  if (hContext) {
    checkCudaErrors(cuCtxDestroy(hContext));
    hContext = 0;
  }

  return 0;
}
