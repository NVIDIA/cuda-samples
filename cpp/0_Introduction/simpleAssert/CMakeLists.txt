cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(simpleAssert LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

# Removes -DNDEBUG For Print specific logs in this sample.
string(REPLACE "-DNDEBUG" "" CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")

include_directories(../../../Common)

add_executable(simpleAssert simpleAssert.cu)

target_compile_options(simpleAssert PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleAssert PRIVATE cxx_std_17 cuda_std_17)

include(InstallSamples)
setup_samples_install()
