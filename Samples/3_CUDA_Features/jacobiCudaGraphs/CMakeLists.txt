cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(jacobiCudaGraphs LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "native")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for jacobiCudaGraphs
add_executable(jacobiCudaGraphs jacobi.cu main.cpp)

target_compile_options(jacobiCudaGraphs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(jacobiCudaGraphs PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(jacobiCudaGraphs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(jacobiCudaGraphs PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)
