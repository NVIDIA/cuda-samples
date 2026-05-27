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
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

inline CompiledKernel compileFileWithNVCC(const char *filename,
                                          int sm_value,
                                          int block_m, int block_n, int block_k,
                                          const std::vector<std::string>& extra_flags) {
    // Check CUDA include path for cuda_fp16.h
    const char *include_path = CUDA_INCLUDE_PATH;
    if (include_path[0] == '\0') {
      printf("\n ERROR: unable to locate CUDA include directory containing cuda_fp16.h\n");
      exit(EXIT_FAILURE);
    }

    std::filesystem::path keep_dir = makeTempPath("matmul_nvcc_keep", "");
    std::error_code ec;
    if (!std::filesystem::create_directory(keep_dir, ec)) {
        std::cerr << "\nerror: unable to create " << keep_dir.string()
                  << " (" << ec.message() << ")\n";
        exit(EXIT_FAILURE);
    }

    std::string base = baseNameWithoutExtension(filename);
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    const char *object_suffix = ".obj";
#else
    const char *object_suffix = ".o";
#endif
    std::filesystem::path object_file = keep_dir / (base + object_suffix);
    std::filesystem::path tile_cubin_file = keep_dir / (base + ".tile.cubin");

    std::vector<std::string> args = {
        NVCC_PATH,
        "--enable-tile",
        "-std=c++20",
        "-arch=sm_" + std::to_string(sm_value),
        "-lineinfo",
        "-c",
        filename,
        "-o",
        object_file.string(),
        "--keep",
        "--keep-dir",
        keep_dir.string(),
        "-I",
        include_path
    };
    appendTileBlockMacroOptions(args, block_m, block_n, block_k);

    for (const auto& flag : extra_flags) {
        args.push_back(flag);
    }

    std::string cmd = joinShellCommand(args);
    std::cerr << "\nCompiling file with NVCC\n";
    int ret = system(cmd.c_str());
    if (ret != 0) {
        fprintf(stderr, "Error: nvcc compilation failed with code %d\n", ret);
        fprintf(stderr, "Command: %s\n", cmd.c_str());
        std::filesystem::remove_all(keep_dir);
        exit(EXIT_FAILURE);
    }

    if (!std::filesystem::exists(tile_cubin_file)) {
        fprintf(stderr, "Error: nvcc did not produce expected Tile cubin %s\n",
                tile_cubin_file.string().c_str());
        std::filesystem::remove_all(keep_dir);
        exit(EXIT_FAILURE);
    }

    CompiledKernel kernel;
    kernel.image = readBinaryFile(tile_cubin_file.string());
    std::filesystem::remove_all(keep_dir);
    return kernel;
}
