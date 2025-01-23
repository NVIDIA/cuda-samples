cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(conjugateGradientMultiDeviceCG LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 89 90 100 120)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for conjugateGradientMultiDeviceCG
add_executable(conjugateGradientMultiDeviceCG conjugateGradientMultiDeviceCG.cu)

target_compile_options(conjugateGradientMultiDeviceCG PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(conjugateGradientMultiDeviceCG PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(conjugateGradientMultiDeviceCG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(conjugateGradientMultiDeviceCG PRIVATE
    CUDA::cublas
    CUDA::cusparse
)
