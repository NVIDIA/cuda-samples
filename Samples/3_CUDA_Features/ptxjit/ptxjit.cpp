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
 * This sample uses the Driver API to just-in-time compile (JIT) a Kernel from
 * PTX code. Additionally, this sample demonstrates the seamless
 * interoperability capability of CUDA runtime Runtime and CUDA Driver API
 * calls. This sample requires Compute Capability 2.0 and higher.
 *
 */

// System includes
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

// CUDA driver & runtime
#include <cuda.h>
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#define CUDA_DRIVER_API
#include <helper_cuda.h>
#include <helper_cuda_drvapi.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

#if defined(_WIN64) || defined(__LP64__)
#define PTX_FILE "ptxjit_kernel64.ptx"
#else
#define PTX_FILE "ptxjit_kernel32.ptx"
#endif

const char *sSDKname = "PTX Just In Time (JIT) Compilation (no-qatest)";

bool inline findModulePath(const char *module_file, std::string &module_path,
                           char **argv, std::string &ptx_source) {
  char *actual_path = sdkFindFilePath(module_file, argv[0]);

  if (actual_path) {
    module_path = actual_path;
  } else {
    printf("> findModulePath file not found: <%s> \n", module_file);
    return false;
  }

  if (module_path.empty()) {
    printf("> findModulePath file not found: <%s> \n", module_file);
    return false;
  } else {
    printf("> findModulePath <%s>\n", module_path.c_str());

    if (module_path.rfind(".ptx") != std::string::npos) {
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

void ptxJIT(int argc, char **argv, CUmodule *phModule, CUfunction *phKernel,
            CUlinkState *lState) {
  CUjit_option options[6];
  void *optionVals[6];
  float walltime;
  char error_log[8192], info_log[8192];
  unsigned int logSize = 8192;
  void *cuOut;
  size_t outSize;
  int myErr = 0;
  std::string module_path, ptx_source;

  // Setup linker options
  // Return walltime from JIT compilation
  options[0] = CU_JIT_WALL_TIME;
  optionVals[0] = (void *)&walltime;
  // Pass a buffer for info messages
  options[1] = CU_JIT_INFO_LOG_BUFFER;
  optionVals[1] = (void *)info_log;
  // Pass the size of the info buffer
  options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  optionVals[2] = (void *)(long)logSize;
  // Pass a buffer for error message
  options[3] = CU_JIT_ERROR_LOG_BUFFER;
  optionVals[3] = (void *)error_log;
  // Pass the size of the error buffer
  options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  optionVals[4] = (void *)(long)logSize;
  // Make the linker verbose
  options[5] = CU_JIT_LOG_VERBOSE;
  optionVals[5] = (void *)1;

  // Create a pending linker invocation
  checkCudaErrors(cuLinkCreate(6, options, optionVals, lState));

  // first search for the module path before we load the results
  if (!findModulePath(PTX_FILE, module_path, argv, ptx_source)) {
    printf("> findModulePath could not find <ptxjit_kernel> ptx\n");
    exit(EXIT_FAILURE);
  } else {
    printf("> initCUDA loading module: <%s>\n", module_path.c_str());
  }

  // Load the PTX from the ptx file
  printf("Loading ptxjit_kernel[] program\n");
  myErr = cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void *)ptx_source.c_str(),
                        strlen(ptx_source.c_str()) + 1, 0, 0, 0, 0);

  if (myErr != CUDA_SUCCESS) {
    // Errors will be put in error_log, per CU_JIT_ERROR_LOG_BUFFER option
    // above.
    fprintf(stderr, "PTX Linker Error:\n%s\n", error_log);
  }

  // Complete the linker step
  checkCudaErrors(cuLinkComplete(*lState, &cuOut, &outSize));

  // Linker walltime and info_log were requested in options above.
  printf("CUDA Link Completed in %fms. Linker Output:\n%s\n", walltime,
         info_log);

  // Load resulting cuBin into module
  checkCudaErrors(cuModuleLoadData(phModule, cuOut));

  // Locate the kernel entry poin
  checkCudaErrors(cuModuleGetFunction(phKernel, *phModule, "myKernel"));

  // Destroy the linker invocation
  checkCudaErrors(cuLinkDestroy(*lState));
}

int main(int argc, char **argv) {
  const unsigned int nThreads = 256;
  const unsigned int nBlocks = 64;
  const size_t memSize = nThreads * nBlocks * sizeof(int);

  CUmodule hModule = 0;
  CUfunction hKernel = 0;
  CUlinkState lState;
  int *d_data = 0;
  int *h_data = 0;

  int cuda_device = 0;

  printf("[%s] - Starting...\n", sSDKname);

  CUdevice dev = findCudaDeviceDRV(argc, (const char **)argv);
  int driverVersion;
  cudaDriverGetVersion(&driverVersion);
  if (driverVersion < CUDART_VERSION) {
    printf(
        "driverVersion = %d < CUDART_VERSION = %d \n"
        "Enhanced compatibility is not supported for this sample.. waving "
        "execution\n",
        driverVersion, CUDART_VERSION);
    exit(EXIT_WAIVED);
  }

  // Allocate memory on host and device (Runtime API)
  // NOTE: The runtime API will create the GPU Context implicitly here
  if ((h_data = (int *)malloc(memSize)) == NULL) {
    std::cerr << "Could not allocate host memory" << std::endl;
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaMalloc(&d_data, memSize));

  // JIT Compile the Kernel from PTX and get the Handles (Driver API)
  ptxJIT(argc, argv, &hModule, &hKernel, &lState);

  // Set the kernel parameters (Driver API)
  dim3 block(nThreads, 1, 1);
  dim3 grid(nBlocks, 1, 1);

  void *args[1] = {&d_data};

  // Launch the kernel (Driver API_)
  checkCudaErrors(cuLaunchKernel(hKernel, grid.x, grid.y, grid.z, block.x,
                                 block.y, block.z, 0, NULL, args, NULL));
  std::cout << "CUDA kernel launched" << std::endl;

  // Copy the result back to the host
  checkCudaErrors(cudaMemcpy(h_data, d_data, memSize, cudaMemcpyDeviceToHost));

  // Check the result
  bool dataGood = true;

  for (unsigned int i = 0; dataGood && i < nBlocks * nThreads; i++) {
    if (h_data[i] != (int)i) {
      std::cerr << "Error at " << i << std::endl;
      dataGood = false;
    }
  }

  // Cleanup
  if (d_data) {
    checkCudaErrors(cudaFree(d_data));
    d_data = 0;
  }

  if (h_data) {
    free(h_data);
    h_data = 0;
  }

  if (hModule) {
    checkCudaErrors(cuModuleUnload(hModule));
    hModule = 0;
  }

  return dataGood ? EXIT_SUCCESS : EXIT_FAILURE;
}
