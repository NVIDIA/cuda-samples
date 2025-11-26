cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(matrixMul_nvrtc LANGUAGES C CXX CUDA)

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
# Add sample target executable
add_executable(matrixMul_nvrtc matrixMul.cpp)

target_compile_options(matrixMul_nvrtc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(matrixMul_nvrtc PRIVATE cxx_std_17 cuda_std_17)

target_link_libraries(matrixMul_nvrtc PRIVATE
    CUDA::nvrtc
    CUDA::cuda_driver
)

# The primary directory of CUDAToolkit_INCLUDE_DIRS is the CUDA Toolkit's include directory for finding the header files.
list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_INCLUDE_DIR)

# Copy clock_kernel.cu to the output directory
add_custom_command(TARGET matrixMul_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/matrixMul_kernel.cu ${CUDA_INCLUDE_DIR}/cooperative_groups.h ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(TARGET matrixMul_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CUDA_INCLUDE_DIR}/cooperative_groups ${CMAKE_CURRENT_BINARY_DIR}/cooperative_groups
)

add_custom_command(TARGET matrixMul_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CUDA_INCLUDE_DIR}/cccl/nv ${CMAKE_CURRENT_BINARY_DIR}/nv
)

add_custom_command(TARGET matrixMul_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CUDA_INCLUDE_DIR}/cccl/cuda ${CMAKE_CURRENT_BINARY_DIR}/cuda
)
