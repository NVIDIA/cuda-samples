cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(StreamPriorities LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Source file
    # Add target for StreamPriorities
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        message(STATUS "Will not build sample streamPriorities - not supported on aarch64")
    else()
        add_executable(StreamPriorities StreamPriorities.cu)

        target_compile_options(StreamPriorities PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(StreamPriorities PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(StreamPriorities PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
else()
    message(STATUS "Will not build sample StreamPriorities - requires Linux OS")
endif()
