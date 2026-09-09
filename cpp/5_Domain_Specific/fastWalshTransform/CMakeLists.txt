cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(fastWalshTransform LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(fastWalshTransform fastWalshTransform.cu fastWalshTransform_gold.cpp)

target_compile_options(fastWalshTransform PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(fastWalshTransform PRIVATE cxx_std_17 cuda_std_17)

include(InstallSamples)
setup_samples_install()
