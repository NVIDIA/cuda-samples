cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(cdpQuadtree LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    # Need to differentiate Tegra_aarch64 and other aarch64 systems(sbsa_aarch64) as they have different CUDA_ARCHITECTURES list
    if(${BUILD_TEGRA})
        set(CMAKE_CUDA_ARCHITECTURES 87 110)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 90)
    endif()
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
# Add target for cdpQuadtree
add_executable(cdpQuadtree cdpQuadtree.cu)

target_compile_options(cdpQuadtree PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cdpQuadtree PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(cdpQuadtree PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
