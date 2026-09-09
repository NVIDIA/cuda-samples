cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(StreamPriorities LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        message(STATUS "Will not build sample streamPriorities - not supported on aarch64")
    else()
        add_executable(StreamPriorities StreamPriorities.cu)
        target_compile_options(StreamPriorities PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        target_compile_features(StreamPriorities PRIVATE cxx_std_17 cuda_std_17)
        include(InstallSamples)
        setup_samples_install()
    endif()
else()
    message(STATUS "Will not build sample StreamPriorities - requires Linux OS")
endif()
