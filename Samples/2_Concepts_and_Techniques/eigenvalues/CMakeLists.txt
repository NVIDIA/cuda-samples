cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(eigenvalues LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for eigenvalues
add_executable(eigenvalues bisect_large.cu bisect_small.cu bisect_util.cu gerschgorin.cpp main.cu matlab.cpp)

target_compile_options(eigenvalues PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(eigenvalues PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(eigenvalues PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Copy data files to output directory
add_custom_command(TARGET eigenvalues POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/diagonal.dat
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Copy data files to output directory
add_custom_command(TARGET eigenvalues POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/superdiagonal.dat
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Copy data files to output directory
add_custom_command(TARGET eigenvalues POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/reference.dat
    ${CMAKE_CURRENT_BINARY_DIR}/
)
