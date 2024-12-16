cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(matrixMul_nvrtc LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "native")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    matrixMul.cpp
)

# Add sample target executable
add_executable(matrixMul_nvrtc ${SRC_FILES})
target_link_libraries(matrixMul_nvrtc PRIVATE
    CUDA::nvrtc
)

# Copy clock_kernel.cu to the output directory
add_custom_command(TARGET matrixMul_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/matrixMul_kernel.cu ${CUDAToolkit_BIN_DIR}/../include/cooperative_groups.h ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(TARGET matrixMul_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CUDAToolkit_BIN_DIR}/../include/cooperative_groups ${CMAKE_CURRENT_BINARY_DIR}/cooperative_groups
)

add_custom_command(TARGET matrixMul_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CUDAToolkit_BIN_DIR}/../include/nv ${CMAKE_CURRENT_BINARY_DIR}/nv
)

add_custom_command(TARGET matrixMul_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CUDAToolkit_BIN_DIR}/../include/cuda ${CMAKE_CURRENT_BINARY_DIR}/cuda
)
