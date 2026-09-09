cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(streamOrderedAllocationP2P LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(streamOrderedAllocationP2P streamOrderedAllocationP2P.cu)

target_compile_options(streamOrderedAllocationP2P PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(streamOrderedAllocationP2P PRIVATE cxx_std_17 cuda_std_17)

include(InstallSamples)
setup_samples_install()
