cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(conjugateGradientMultiBlockCG LANGUAGES CUDA CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

# This sample is not supported on QNX
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    message(STATUS "Will not build sample ${PROJECT_NAME} - not supported on QNX")
    return()
endif()

add_executable(conjugateGradientMultiBlockCG conjugateGradientMultiBlockCG.cu)

target_compile_options(conjugateGradientMultiBlockCG PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(conjugateGradientMultiBlockCG PRIVATE cxx_std_17 cuda_std_17)

target_link_libraries(conjugateGradientMultiBlockCG PRIVATE
    CUDA::cublas
    CUDA::cusparse
)

include(InstallSamples)
setup_samples_install()
