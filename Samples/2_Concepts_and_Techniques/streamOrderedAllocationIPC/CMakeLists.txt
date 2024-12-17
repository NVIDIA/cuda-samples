cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(streamOrderedAllocationIPC LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for streamOrderedAllocationIPC
add_executable(streamOrderedAllocationIPC streamOrderedAllocationIPC.cu ../../../Common/helper_multiprocess.cpp)

target_compile_options(streamOrderedAllocationIPC PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(streamOrderedAllocationIPC PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(streamOrderedAllocationIPC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(streamOrderedAllocationIPC PUBLIC
    CUDA::cuda_driver
)
