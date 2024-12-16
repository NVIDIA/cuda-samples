cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(simpleSurfaceWrite LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "native")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for simpleSurfaceWrite
add_executable(simpleSurfaceWrite simpleSurfaceWrite.cu)

target_compile_options(simpleSurfaceWrite PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleSurfaceWrite PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(simpleSurfaceWrite PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Copy data to the output directory
add_custom_command(TARGET simpleSurfaceWrite POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_BINARY_DIR}
)
