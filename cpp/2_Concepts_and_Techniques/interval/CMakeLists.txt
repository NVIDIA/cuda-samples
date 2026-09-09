cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(interval LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(interval interval.cu)

target_compile_options(interval PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(interval PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(interval PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

include(InstallSamples)
setup_samples_install()
