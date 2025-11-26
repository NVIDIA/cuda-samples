cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(MersenneTwisterGP11213 LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for MersenneTwisterGP11213
add_executable(MersenneTwisterGP11213 MersenneTwister.cpp)

target_compile_options(MersenneTwisterGP11213 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(MersenneTwisterGP11213 PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(MersenneTwisterGP11213 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(MersenneTwisterGP11213 PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(MersenneTwisterGP11213 PRIVATE
    CUDA::cudart
    CUDA::curand
)
