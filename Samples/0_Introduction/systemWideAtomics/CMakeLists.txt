cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(systemWideAtomics LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 60 61 70 72 75 80 86 87 89 90 100 101 120)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS "Will not build sample systemWideAtomics - not supported on aarch64")
else()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Source file
        # Add target for systemWideAtomics
        add_executable(systemWideAtomics systemWideAtomics.cu)

        target_compile_options(systemWideAtomics PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(systemWideAtomics PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(systemWideAtomics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    else()
        message(STATUS "Will not build sample systemWideAtomics - requires Linux OS")
    endif()
endif()
