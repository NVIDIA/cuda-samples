cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(dct8x8 LANGUAGES C CXX CUDA)

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
# Add target for dct8x8
add_executable(dct8x8 dct8x8.cu BmpUtil.cpp DCT8x8_Gold.cpp)

target_compile_options(dct8x8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(dct8x8 PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(dct8x8 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(dct8x8 PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)

file(GLOB REF_DATA "teapot512*")
add_custom_command(TARGET dct8x8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${REF_DATA}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Copy data files to output directory
add_custom_command(TARGET dct8x8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/teapot512.ppm
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Copy data files to output directory
add_custom_command(TARGET dct8x8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/teapot512.bmp
    ${CMAKE_CURRENT_BINARY_DIR}/
)
