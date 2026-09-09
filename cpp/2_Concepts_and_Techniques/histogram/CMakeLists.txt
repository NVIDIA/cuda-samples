cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(histogram LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(histogram histogram256.cu histogram64.cu histogram_gold.cpp main.cpp)

target_compile_options(histogram PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(histogram PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(histogram PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)

include(InstallSamples)
setup_samples_install()
