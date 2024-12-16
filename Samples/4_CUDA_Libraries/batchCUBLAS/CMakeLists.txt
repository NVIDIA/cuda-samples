cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(batchCUBLAS LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "native")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for batchCUBLAS
add_executable(batchCUBLAS batchCUBLAS.cpp)

target_compile_options(batchCUBLAS PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(batchCUBLAS PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(batchCUBLAS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(batchCUBLAS PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(batchCUBLAS PRIVATE
    CUDA::cublas
    CUDA::cudart
)
