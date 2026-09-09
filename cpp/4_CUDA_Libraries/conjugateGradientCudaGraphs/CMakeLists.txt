cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(conjugateGradientCudaGraphs LANGUAGES CUDA CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

# Disable response file for libraries on QNX as qcc does not support lib paths with double quotes
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
endif()

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(../../../Common)

add_executable(conjugateGradientCudaGraphs conjugateGradientCudaGraphs.cu)

target_compile_options(conjugateGradientCudaGraphs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(conjugateGradientCudaGraphs PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(conjugateGradientCudaGraphs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(conjugateGradientCudaGraphs PRIVATE
    CUDA::cublas
    CUDA::cusparse
)

include(InstallSamples)
setup_samples_install()
