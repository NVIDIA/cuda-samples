cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(simpleVoteIntrinsics LANGUAGES C CXX CUDA)

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
    simpleVoteIntrinsics.cu
)

# Add target for simpleVoteIntrinsics
add_executable(simpleVoteIntrinsics ${SRC_FILES})
set_target_properties(simpleVoteIntrinsics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
