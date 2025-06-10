cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(cdpSimpleQuicksort LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The aarch64/sbsa_aarch64 CUDA toolkit are support on Tegra since 13.0, so need to check which version of the toolkit is installed
string(FIND "${CUDAToolkit_INCLUDE_DIRS}" "aarch64-linux" _aarch64_linux_ctk)
string(FIND "${CUDAToolkit_INCLUDE_DIRS}" "aarch64-qnx" _aarch64_qnx_ctk)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" AND (NOT _aarch64_linux_ctk EQUAL -1 OR NOT _aarch64_qnx_ctk EQUAL -1))
    set(CMAKE_CUDA_ARCHITECTURES 87 110)
else()
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 100 110 120)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for cdpSimpleQuicksort
add_executable(cdpSimpleQuicksort cdpSimpleQuicksort.cu)

target_compile_options(cdpSimpleQuicksort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cdpSimpleQuicksort PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(cdpSimpleQuicksort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
