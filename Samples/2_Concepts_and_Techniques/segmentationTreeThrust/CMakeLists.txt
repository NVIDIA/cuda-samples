cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(segmentationTreeThrust LANGUAGES C CXX CUDA)

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
# Add target for segmentationTreeThrust
add_executable(segmentationTreeThrust segmentationTree.cu)

target_compile_options(segmentationTreeThrust PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(segmentationTreeThrust PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(segmentationTreeThrust PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Copy data files to output directory
add_custom_command(TARGET segmentationTreeThrust POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/test.ppm
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Copy data files to output directory
add_custom_command(TARGET segmentationTreeThrust POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_00.ppm
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Copy data files to output directory
add_custom_command(TARGET segmentationTreeThrust POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_09.ppm
    ${CMAKE_CURRENT_BINARY_DIR}/
)
