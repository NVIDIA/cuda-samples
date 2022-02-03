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

#ifndef COMMON_NVRTC_HELPER_H_

#define COMMON_NVRTC_HELPER_H_ 1

#include <cuda.h>
#include <helper_cuda_drvapi.h>
#include <nvrtc.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define NVRTC_SAFE_CALL(Name, x)                                \
  do {                                                          \
    nvrtcResult result = x;                                     \
    if (result != NVRTC_SUCCESS) {                              \
      std::cerr << "\nerror: " << Name << " failed with error " \
                << nvrtcGetErrorString(result);                 \
      exit(1);                                                  \
    }                                                           \
  } while (0)

void compileFileToCUBIN(char *filename, int argc, char **argv, char **cubinResult,
                      size_t *cubinResultSize, int requiresCGheaders) {
  if (!filename) {
    std::cerr << "\nerror: filename is empty for compileFileToCUBIN()!\n";
    exit(1);
  }

  std::ifstream inputFile(filename,
                          std::ios::in | std::ios::binary | std::ios::ate);

  if (!inputFile.is_open()) {
    std::cerr << "\nerror: unable to open " << filename << " for reading!\n";
    exit(1);
  }

  std::streampos pos = inputFile.tellg();
  size_t inputSize = (size_t)pos;
  char *memBlock = new char[inputSize + 1];

  inputFile.seekg(0, std::ios::beg);
  inputFile.read(memBlock, inputSize);
  inputFile.close();
  memBlock[inputSize] = '\x0';

  int numCompileOptions = 0;

  char *compileParams[2];

  int major = 0, minor = 0;
  char deviceName[256];

  // Picks the best CUDA device available
  CUdevice cuDevice = findCudaDeviceDRV(argc, (const char **)argv);

  // get compute capabilities and the devicename
  checkCudaErrors(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  checkCudaErrors(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
  
  {
  // Compile cubin for the GPU arch on which are going to run cuda kernel.
  std::string compileOptions;
  compileOptions = "--gpu-architecture=sm_";

  compileParams[numCompileOptions] = reinterpret_cast<char *>(
                  malloc(sizeof(char) * (compileOptions.length() + 10)));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(compileParams[numCompileOptions], sizeof(char) * (compileOptions.length() + 10),
            "%s%d%d", compileOptions.c_str(), major, minor);
#else
  snprintf(compileParams[numCompileOptions], compileOptions.size() + 10, "%s%d%d",
           compileOptions.c_str(), major, minor);
#endif
  }

  numCompileOptions++;

  if (requiresCGheaders) {
    std::string compileOptions;
    char HeaderNames[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(HeaderNames, sizeof(HeaderNames), "%s", "cooperative_groups.h");
#else
    snprintf(HeaderNames, sizeof(HeaderNames), "%s", "cooperative_groups.h");
#endif

    compileOptions = "--include-path=";

    char *strPath = sdkFindFilePath(HeaderNames, argv[0]);
    if (!strPath) {
      std::cerr << "\nerror: header file " << HeaderNames << " not found!\n";
      exit(1);
    }
    std::string path = strPath;
    if (!path.empty()) {
      std::size_t found = path.find(HeaderNames);
      path.erase(found);
    } else {
      printf(
          "\nCooperativeGroups headers not found, please install it in %s "
          "sample directory..\n Exiting..\n",
          argv[0]);
      exit(1);
    }
    compileOptions += path.c_str();
    compileParams[numCompileOptions] = reinterpret_cast<char *>(
        malloc(sizeof(char) * (compileOptions.length() + 1)));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(compileParams[numCompileOptions], sizeof(char) * (compileOptions.length() + 1),
              "%s", compileOptions.c_str());
#else
    snprintf(compileParams[numCompileOptions], compileOptions.size(), "%s",
             compileOptions.c_str());
#endif
    numCompileOptions++;
  }

  // compile
  nvrtcProgram prog;
  NVRTC_SAFE_CALL("nvrtcCreateProgram",
                  nvrtcCreateProgram(&prog, memBlock, filename, 0, NULL, NULL));

  nvrtcResult res = nvrtcCompileProgram(prog, numCompileOptions, compileParams);

  // dump log
  size_t logSize;
  NVRTC_SAFE_CALL("nvrtcGetProgramLogSize",
                  nvrtcGetProgramLogSize(prog, &logSize));
  char *log = reinterpret_cast<char *>(malloc(sizeof(char) * logSize + 1));
  NVRTC_SAFE_CALL("nvrtcGetProgramLog", nvrtcGetProgramLog(prog, log));
  log[logSize] = '\x0';

  if (strlen(log) >= 2) {
    std::cerr << "\n compilation log ---\n";
    std::cerr << log;
    std::cerr << "\n end log ---\n";
  }

  free(log);

  NVRTC_SAFE_CALL("nvrtcCompileProgram", res);

  size_t codeSize;
  NVRTC_SAFE_CALL("nvrtcGetCUBINSize", nvrtcGetCUBINSize(prog, &codeSize));
  char *code = new char[codeSize];
  NVRTC_SAFE_CALL("nvrtcGetCUBIN", nvrtcGetCUBIN(prog, code));
  *cubinResult = code;
  *cubinResultSize = codeSize;

  for (int i = 0; i < numCompileOptions; i++) {
    free(compileParams[i]);
  }
}

CUmodule loadCUBIN(char *cubin, int argc, char **argv) {
  CUmodule module;
  CUcontext context;
  int major = 0, minor = 0;
  char deviceName[256];

  // Picks the best CUDA device available
  CUdevice cuDevice = findCudaDeviceDRV(argc, (const char **)argv);

  // get compute capabilities and the devicename
  checkCudaErrors(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  checkCudaErrors(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
  checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
  printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuCtxCreate(&context, 0, cuDevice));

  checkCudaErrors(cuModuleLoadData(&module, cubin));
  free(cubin);

  return module;
}

#endif  // COMMON_NVRTC_HELPER_H_
