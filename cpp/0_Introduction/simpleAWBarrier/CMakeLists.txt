cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(simpleAWBarrier LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

# This sample is not supported on QNX
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    message(STATUS "simpleAWBarrier is not supported on QNX")
    return()
endif()

add_executable(simpleAWBarrier simpleAWBarrier.cu)

target_compile_options(simpleAWBarrier PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleAWBarrier PRIVATE cxx_std_17 cuda_std_17)

include(InstallSamples)
setup_samples_install()
