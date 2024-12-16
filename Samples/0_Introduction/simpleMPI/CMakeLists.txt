cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(simpleMPI LANGUAGES C CXX CUDA)

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

find_package(MPI)

# Source file
set(SRC_FILES
    simpleMPI.cpp
    simpleMPI.cu
)

if(${MPI_FOUND})
    # Add target for simpleMPI
    add_executable(simpleMPI ${SRC_FILES})
    set_target_properties(simpleMPI PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(simpleMPI PUBLIC
        ${MPI_INCLUDE_PATH}
    )

    target_link_libraries(simpleMPI PUBLIC
        ${MPI_CXX_LIBRARIES}
    )

else()
    message(STATUS "MPI not found - will not build sample 'simpleMPI'")
endif()
