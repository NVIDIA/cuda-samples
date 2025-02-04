cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(cudaOpenMP LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

find_package(OpenMP)

# Source file
if(OpenMP_CXX_FOUND)
# Add target for asyncAPI
    add_executable(cudaOpenMP cudaOpenMP.cu)

target_compile_options(cudaOpenMP PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cudaOpenMP PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(cudaOpenMP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(cudaOpenMP PUBLIC
        OpenMP::OpenMP_CXX
    )
else()
    message(STATUS "OpenMP not found - will not build sample 'cudaOpenMP'")
endif()
