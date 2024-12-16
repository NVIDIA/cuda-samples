cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(simpleAtomicIntrinsics_nvrtc LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "native")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add sample target executable
add_executable(simpleAtomicIntrinsics_nvrtc simpleAtomicIntrinsics_cpu.cpp simpleAtomicIntrinsics.cpp)

target_compile_options(simpleAtomicIntrinsics_nvrtc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleAtomicIntrinsics_nvrtc PRIVATE cxx_std_17 cuda_std_17)

target_link_libraries(simpleAtomicIntrinsics_nvrtc PRIVATE
    CUDA::nvrtc
)

# Copy clock_kernel.cu to the output directory
add_custom_command(TARGET simpleAtomicIntrinsics_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/simpleAtomicIntrinsics_kernel.cuh
    ${CMAKE_CURRENT_BINARY_DIR}
)
