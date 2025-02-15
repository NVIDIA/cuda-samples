cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(bf16TensorCoreGemm LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86 87 89 90 100 101 120)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for bf16TensorCoreGemm
add_executable(bf16TensorCoreGemm bf16TensorCoreGemm.cu)

target_compile_options(bf16TensorCoreGemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(bf16TensorCoreGemm PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(bf16TensorCoreGemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
