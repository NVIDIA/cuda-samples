cmake_minimum_required(VERSION 3.20)

set(SAMPLE_DISALLOW_ARCHS 75)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(tileMatmul LANGUAGES C CXX CUDA)

# Tile samples need C++20. If the host or CUDA compiler can't provide it,
# skip this sample instead of aborting configure.
if(NOT "cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES
   OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
   OR NOT "cuda_std_20" IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
    message(STATUS "Compiler lacks required C++20 support - skipping ${PROJECT_NAME}")
    return()
endif()

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --enable-tile")

if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common ../Benchmark_Common)

# Source file
add_executable(tileMatmul tileMatmul.cu)

target_compile_features(tileMatmul PRIVATE cxx_std_20 cuda_std_20)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
