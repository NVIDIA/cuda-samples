cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(simpleGLES_EGLOutput LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Source file
    # Add target for simpleGLES_EGLOutput
    add_executable(simpleGLES_EGLOutput simpleGLES_EGLOutput.cu)

    target_compile_options(simpleGLES_EGLOutput PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(simpleGLES_EGLOutput PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(simpleGLES_EGLOutput PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    message(STATUS "Will not build sample simpleGLES_EGLOutput - requires Linux OS")
endif()
