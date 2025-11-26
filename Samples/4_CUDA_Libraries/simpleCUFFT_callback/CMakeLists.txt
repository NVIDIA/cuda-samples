cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleCUFFT_callback LANGUAGES CUDA)

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

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Source file
    # Add target for simpleCUFFT_callback
    add_executable(simpleCUFFT_callback simpleCUFFT_callback.cu)

    target_compile_options(simpleCUFFT_callback PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(simpleCUFFT_callback PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(simpleCUFFT_callback PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_link_libraries(simpleCUFFT_callback PRIVATE
        CUDA::cufft_static
        culibos
    )
else()
    message(STATUS "Will not build sample simpleCUFFT_callback - requires Linux OS")
endif()
