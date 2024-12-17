cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(simpleCUFFT_2d_MGPU LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for simpleCUFFT_2d_MGPU
add_executable(simpleCUFFT_2d_MGPU simpleCUFFT_2d_MGPU.cu)

target_compile_options(simpleCUFFT_2d_MGPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleCUFFT_2d_MGPU PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(simpleCUFFT_2d_MGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(simpleCUFFT_2d_MGPU PRIVATE
    CUDA::cufft
)
