cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(simpleCUBLASXT LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "native")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for simpleCUBLASXT
add_executable(simpleCUBLASXT simpleCUBLASXT.cpp)

target_compile_options(simpleCUBLASXT PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleCUBLASXT PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(simpleCUBLASXT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(simpleCUBLASXT PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(simpleCUBLASXT PRIVATE
    CUDA::cudart
    CUDA::cublas
)
