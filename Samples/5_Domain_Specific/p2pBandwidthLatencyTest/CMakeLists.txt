cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(p2pBandwidthLatencyTest LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for p2pBandwidthLatencyTest
add_executable(p2pBandwidthLatencyTest p2pBandwidthLatencyTest.cu)

target_compile_options(p2pBandwidthLatencyTest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(p2pBandwidthLatencyTest PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(p2pBandwidthLatencyTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
