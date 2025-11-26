cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(conjugateGradientCudaGraphs LANGUAGES CUDA CXX)

# Disable response file for libraries on QNX as qcc does not support lib paths with double quotes
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
endif()

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

# Source file
# Add target for conjugateGradientCudaGraphs
add_executable(conjugateGradientCudaGraphs conjugateGradientCudaGraphs.cu)

target_compile_options(conjugateGradientCudaGraphs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(conjugateGradientCudaGraphs PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(conjugateGradientCudaGraphs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(conjugateGradientCudaGraphs PRIVATE
    CUDA::cublas
    CUDA::cusparse
)
