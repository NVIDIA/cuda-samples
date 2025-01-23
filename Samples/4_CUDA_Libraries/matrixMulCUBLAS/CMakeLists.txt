cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(matrixMulCUBLAS LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for matrixMulCUBLAS
add_executable(matrixMulCUBLAS matrixMulCUBLAS.cpp)

target_compile_options(matrixMulCUBLAS PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(matrixMulCUBLAS PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(matrixMulCUBLAS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(matrixMulCUBLAS PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(matrixMulCUBLAS PRIVATE
    CUDA::cudart
    CUDA::cublas
)
