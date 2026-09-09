cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(simpleLayeredTexture LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(simpleLayeredTexture simpleLayeredTexture.cu)

target_compile_options(simpleLayeredTexture PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleLayeredTexture PRIVATE cxx_std_17 cuda_std_17)

include(InstallSamples)
setup_samples_install()
