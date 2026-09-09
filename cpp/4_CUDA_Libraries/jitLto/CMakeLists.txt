cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(jitLto LANGUAGES CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

# This sample is not supported on QNX
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    message(STATUS "Will not build sample ${PROJECT_NAME} - not supported on QNX")
    return()
endif()

add_executable(jitLto jitLto.cpp)

target_compile_options(jitLto PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(jitLto PRIVATE cxx_std_17 cuda_std_17)

find_library(NVJITLINK_LIB
             NAME nvJitLink
             PATHS "${CUDAToolkit_LIBRARY_DIR}" "${CMAKE_LIBRARY_PATH}")

target_include_directories(jitLto PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(jitLto PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
    ${NVJITLINK_LIB}
)

include(InstallSamples)
setup_samples_install()
