cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(simpleIPC LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS "Will not build sample simpleIPC - not supported on aarch64")
else()
    add_executable(simpleIPC simpleIPC.cu ../../../Common/helper_multiprocess.cpp)

    target_compile_options(simpleIPC PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(simpleIPC PRIVATE cxx_std_17 cuda_std_17)

endif()

include(InstallSamples)
setup_samples_install()
