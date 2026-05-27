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

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <helper_string.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <process.h>
#else
#include <unistd.h>
#endif

enum class CompilerBackend {
    NVRTC,
    NVCC
};

struct CompiledKernel {
    std::vector<char> image;
};

struct TileConfig {
    int block_m;
    int block_n;
    int block_k;
};

struct SearchSpace {
    std::vector<TileConfig> tile_options;
    std::vector<int> load_latency_options;
    std::vector<int> store_latency_options;
};

static constexpr const char *kSearchSpaceFileName = "autotuner_search_space.conf";

inline const char* compilerBackendName(CompilerBackend compiler_backend) {
    return compiler_backend == CompilerBackend::NVCC ? "NVCC" : "NVRTC";
}

inline int ceilDiv(int a, int b) {
    return (a + b - 1) / b;
}

inline unsigned long getProcessId() {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    return static_cast<unsigned long>(_getpid());
#else
    return static_cast<unsigned long>(getpid());
#endif
}

inline std::string makeTempPath(const char *prefix, const char *suffix) {
    static unsigned int counter = 0;
    std::string filename = std::string(prefix) + "_" + std::to_string(getProcessId()) + "_" +
                           std::to_string(static_cast<long>(time(NULL))) + "_" +
                           std::to_string(counter++) + suffix;
    return (std::filesystem::temp_directory_path() / filename).string();
}

inline std::string shellQuote(const std::string& arg) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    std::string quoted = "\"";
    for (char c : arg) {
        if (c == '"') {
            quoted += "\\\"";
        } else {
            quoted += c;
        }
    }
    quoted += "\"";
    return quoted;
#else
    std::string quoted = "'";
    for (char c : arg) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
#endif
}

inline std::string joinShellCommand(const std::vector<std::string>& args) {
    std::string cmd;
    for (const auto& arg : args) {
        if (!cmd.empty()) {
            cmd += " ";
        }
        cmd += shellQuote(arg);
    }
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    return "\"" + cmd + "\"";
#else
    return cmd;
#endif
}

inline std::vector<char> readBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "\nerror: unable to open " << path << " for reading!\n";
        exit(EXIT_FAILURE);
    }

    std::streamsize size = file.tellg();
    if (size < 0) {
        std::cerr << "\nerror: unable to determine size of " << path << "\n";
        exit(EXIT_FAILURE);
    }

    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (size > 0 && !file.read(data.data(), size)) {
        std::cerr << "\nerror: unable to read " << path << "\n";
        exit(EXIT_FAILURE);
    }
    return data;
}

inline std::string baseNameWithoutExtension(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) {
        base.resize(dot);
    }
    return base;
}

inline void appendTileBlockMacroOptions(std::vector<std::string>& options,
                                        int block_m, int block_n, int block_k) {
    options.push_back("-DTILE_BLOCK_M=" + std::to_string(block_m));
    options.push_back("-DTILE_BLOCK_N=" + std::to_string(block_n));
    options.push_back("-DTILE_BLOCK_K=" + std::to_string(block_k));
}

inline bool parsePositiveInt(const std::string& text, int *value) {
    char *end = nullptr;
    errno = 0;
    long parsed = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' ||
        parsed <= 0 || parsed > INT_MAX) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

inline void searchSpaceError(const char *filename,
                                          int line_number,
                                          const std::string& message) {
    fprintf(stderr, "Error: %s:%d: %s\n", filename, line_number, message.c_str());
    exit(EXIT_FAILURE);
}

inline char *copyFilePath(const std::string& path) {
    char *file_path = reinterpret_cast<char *>(malloc(path.length() + 1));
    if (file_path == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for file path\n");
        exit(EXIT_FAILURE);
    }
    std::memcpy(file_path, path.c_str(), path.length() + 1);
    return file_path;
}

inline char *findSampleFile(const char *filename, const char *executable_path) {
    if (executable_path != NULL) {
        std::filesystem::path executable_dir =
            std::filesystem::path(executable_path).parent_path();
        if (!executable_dir.empty()) {
            std::filesystem::path candidate = executable_dir / filename;
            if (std::filesystem::exists(candidate)) {
                return copyFilePath(candidate.string());
            }
        }
    }

    return sdkFindFilePath(filename, executable_path);
}

inline SearchSpace loadSearchSpace(const char *filename) {
    std::ifstream input(filename);
    if (!input.is_open()) {
        fprintf(stderr, "Error: unable to open search space file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    SearchSpace search_space;
    std::string line;
    int line_number = 0;

    while (std::getline(input, line)) {
        line_number++;
        size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line.resize(comment);
        }

        std::istringstream tokens(line);
        std::string directive;
        if (!(tokens >> directive)) {
            continue;
        }

        std::vector<std::string> values;
        std::string value;
        while (tokens >> value) {
            values.push_back(value);
        }

        if (directive == "tile") {
            if (values.size() != 3) {
                searchSpaceError(filename, line_number,
                                 "tile expects block_m block_n block_k");
            }

            TileConfig tile = {};
            if (!parsePositiveInt(values[0], &tile.block_m) ||
                !parsePositiveInt(values[1], &tile.block_n) ||
                !parsePositiveInt(values[2], &tile.block_k)) {
                searchSpaceError(filename, line_number,
                                 "tile values must be positive integers");
            }
            search_space.tile_options.push_back(tile);
        } else if (directive == "load_latency") {
            if (values.empty()) {
                searchSpaceError(filename, line_number,
                                 "load_latency expects at least one value");
            }
            for (const auto& option : values) {
                int latency = 0;
                if (!parsePositiveInt(option, &latency)) {
                    searchSpaceError(filename, line_number,
                                     "load_latency values must be positive integers");
                }
                search_space.load_latency_options.push_back(latency);
            }
        } else if (directive == "store_latency") {
            if (values.empty()) {
                searchSpaceError(filename, line_number,
                                 "store_latency expects at least one value");
            }
            for (const auto& option : values) {
                int latency = 0;
                if (!parsePositiveInt(option, &latency)) {
                    searchSpaceError(filename, line_number,
                                     "store_latency values must be positive integers");
                }
                search_space.store_latency_options.push_back(latency);
            }
        } else {
            searchSpaceError(filename, line_number,
                             "unknown search space directive '" + directive + "'");
        }
    }

    if (search_space.tile_options.empty()) {
        fprintf(stderr, "Error: search space file %s does not list any tile entries\n", filename);
        exit(EXIT_FAILURE);
    }
    if (search_space.load_latency_options.empty()) {
        fprintf(stderr, "Error: search space file %s does not list any load_latency entries\n",
                filename);
        exit(EXIT_FAILURE);
    }
    if (search_space.store_latency_options.empty()) {
        fprintf(stderr, "Error: search space file %s does not list any store_latency entries\n",
                filename);
        exit(EXIT_FAILURE);
    }

    return search_space;
}
