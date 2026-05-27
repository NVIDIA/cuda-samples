/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include "backend_common.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nvrtc.h>

#define NVRTC_SAFE_CALL(Name, x)                                             \
    do {                                                                     \
        nvrtcResult result = x;                                              \
        if (result != NVRTC_SUCCESS) {                                       \
            std::cerr << "\nerror: " << Name << " failed with error " <<     \
                      nvrtcGetErrorString(result);                           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while(0)

inline std::vector<char> compileTileIRToCubin(const char *tileIR,
                                              size_t tileIRSize,
                                              int sm_value) {
    // Generate unique temporary file names using PID, timestamp, and a local counter.
    std::string tileir_file = makeTempPath("tileir", ".bc");
    std::string cubin_file = makeTempPath("cubin", ".cubin");

    // Write TileIR to a temporary file because tileiras consumes files rather than memory buffers.
    FILE* fp = fopen(tileir_file.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "Error: failed to open %s for writing\n", tileir_file.c_str());
        exit(EXIT_FAILURE);
    }
    fwrite(tileIR, 1, tileIRSize, fp);
    fclose(fp);

    /*
     * Ideally we would use the Driver API to compile TileIR to cubin with the
     * latest driver installed. However, to avoid the hassle of upgrading the
     * driver, we use tileiras for now, which is handily available in CUDA
     * Toolkit.
     *
     * The Driver API path would look like:
     * (requires #include <helper_cuda_drvapi.h>)
     *
     *     CUmodule module;
     *     CUjit_option options[] = {
     *         CU_JIT_GENERATE_LINE_INFO
     *     };
     *     void* optionValues[] = {
     *         (void*)(uintptr_t)1,  // line info
     *     };
     *     unsigned int numOptions = sizeof(options) / sizeof(options[0]);
     *     checkCudaErrors(cuModuleLoadDataEx(&module, tileIR, numOptions,
     *                                        options, optionValues));
     */

    // Compile TileIR to cubin using tileiras from the configured CUDA Toolkit.
    // This happens before benchmarking so the timed path matches the NVCC backend.
    std::string cmd = joinShellCommand({
        TILEIRAS_PATH,
        "-arch=sm_" + std::to_string(sm_value),
        tileir_file,
        "-o",
        cubin_file
    });
    int ret = system(cmd.c_str());
    if (ret != 0) {
        fprintf(stderr, "Error: tileiras compilation failed with code %d\n", ret);
        fprintf(stderr, "Command: %s\n", cmd.c_str());
        remove(tileir_file.c_str());
        remove(cubin_file.c_str());
        exit(EXIT_FAILURE);
    }

    // Read the generated cubin into memory so later benchmark iterations do not depend on temp files.
    std::vector<char> cubin = readBinaryFile(cubin_file);

    // Remove temporary files after the cubin has been captured.
    remove(tileir_file.c_str());
    remove(cubin_file.c_str());
    return cubin;
}

inline CompiledKernel compileFileWithNVRTC(const char *filename,
                                           int sm_value,
                                           int block_m, int block_n, int block_k,
                                           const std::vector<std::string>& extra_flags) {
    // Check include path for cuda_fp16.h
    const char *ptr = CUDA_INCLUDE_PATH;
    if (ptr[0] == '\0') {
      printf("\n ERROR: unable to locate CUDA include directory containing cuda_fp16.h\n");
      exit(EXIT_FAILURE);
    }
    std::vector<std::string> option_storage = {
        "-I",
        ptr,
        "-std=c++20",
        "-enable-tile",
        "-lineinfo",
        "-arch=compute_" + std::to_string(sm_value)
    };
    appendTileBlockMacroOptions(option_storage, block_m, block_n, block_k);
    for (const auto& flag : extra_flags) {
      option_storage.push_back(flag);
    }

    std::vector<const char*> argv_vec;
    for (const auto& option : option_storage) {
      argv_vec.push_back(option.c_str());
    }
    const char **argv = argv_vec.data();
    int argc = static_cast<int>(argv_vec.size());
    std::cerr << "\nCompiling file with NVRTC\n";
    std::ifstream inputFile(filename, std::ios::in | std::ios::binary |
              std::ios::ate);
    if (!inputFile.is_open()) {
    	std::cerr << "\nerror: unable to open " << filename << " for reading!\n";
    	exit(EXIT_FAILURE);
    }
    std::streampos pos = inputFile.tellg();
    size_t inputSize = pos;
    char * memBlock = new char [inputSize + 1];
    inputFile.seekg (0, std::ios::beg);
    inputFile.read (memBlock, inputSize);
    inputFile.close();
    memBlock[inputSize] = '\x0';

    // Compile the source string to PTX and Tile IR.
    nvrtcProgram prog;
    NVRTC_SAFE_CALL("nvrtcCreateProgram", nvrtcCreateProgram(&prog, memBlock,
    "testprog", 0, NULL, NULL));
    nvrtcResult res = nvrtcCompileProgram(prog, argc, argv);
    // Dump the NVRTC compilation log
    size_t logSize;
    NVRTC_SAFE_CALL("nvrtcGetProgramLogSize", nvrtcGetProgramLogSize(prog, &logSize));
    char* log = (char*)malloc(sizeof(char) * logSize + 1);
    NVRTC_SAFE_CALL("nvrtcGetProgramLog", nvrtcGetProgramLog(prog, log));
    log[logSize] = '\x0';
    std::cerr << "\n compilation log ---\n";
    std::cerr << log;
    std::cerr << "\n end log ---\n\n";
    free(log);
    NVRTC_SAFE_CALL("nvrtcCompileProgram", res);

    // Fetch Tile IR and compile it to cubin before benchmarking.
    size_t tileIRSize;
    NVRTC_SAFE_CALL("nvrtcGetTileIRSize", nvrtcGetTileIRSize(prog, &tileIRSize));
    std::vector<char> tileIR(tileIRSize);
    NVRTC_SAFE_CALL("nvrtcGetTileIR", nvrtcGetTileIR(prog, tileIR.data()));
    CompiledKernel kernel;
    kernel.image = compileTileIRToCubin(tileIR.data(), tileIR.size(), sm_value);
    NVRTC_SAFE_CALL("nvrtcDestroyProgram", nvrtcDestroyProgram(&prog));
    delete[] memBlock;
    return kernel;
}
