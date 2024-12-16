cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(simpleCudaGraphs LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "native")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for simpleCudaGraphs
add_executable(simpleCudaGraphs simpleCudaGraphs.cu)

target_compile_options(simpleCudaGraphs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleCudaGraphs PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(simpleCudaGraphs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
