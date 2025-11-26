cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(UnifiedMemoryStreams LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

# This sample is not supported on QNX
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    message(STATUS "Will not build sample ${PROJECT_NAME} - not supported on QNX")
    return()
endif()

# Source file
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    find_package(OpenMP REQUIRED C CXX)
else()
    find_package(OpenMP REQUIRED)
endif()

if(${OpenMP_FOUND})
    # Add target for UnifiedMemoryStreams
    add_executable(UnifiedMemoryStreams UnifiedMemoryStreams.cu)

target_compile_options(UnifiedMemoryStreams PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(UnifiedMemoryStreams PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(UnifiedMemoryStreams PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_link_libraries(UnifiedMemoryStreams PUBLIC
        CUDA::cublas
        OpenMP::OpenMP_CXX
    )
else()
    message(STATUS "OpenMP not found - will not build sample 'UnifiedMemoryStreams'")
endif()
