cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(cdpAdvancedQuicksort LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86 87 90)
else()
    set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for cdpAdvancedQuicksort
add_executable(cdpAdvancedQuicksort cdpAdvancedQuicksort.cu cdpBitonicSort.cu)

target_compile_options(cdpAdvancedQuicksort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cdpAdvancedQuicksort PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(cdpAdvancedQuicksort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
