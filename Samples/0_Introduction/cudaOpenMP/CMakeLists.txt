cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(cudaOpenMP LANGUAGES C CXX CUDA)

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

find_package(OpenMP)

# Source file
set(SRC_FILES
    cudaOpenMP.cu
)

if(OpenMP_CXX_FOUND)
# Add target for asyncAPI
    add_executable(cudaOpenMP ${SRC_FILES})
    set_target_properties(cudaOpenMP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(cudaOpenMP PUBLIC
        OpenMP::OpenMP_CXX
    )
else()
    message(STATUS "OpenMP not found - will not build sample 'cudaOpenMP'")
endif()
