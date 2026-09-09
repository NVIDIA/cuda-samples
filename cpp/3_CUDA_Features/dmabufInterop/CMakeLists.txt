cmake_minimum_required(VERSION 3.20)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(dmabufInterop LANGUAGES C CXX CUDA)

# dma-buf is a Linux-only kernel facility. Skip the sample on all other platforms.
if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    message(STATUS "Skipping dmabufInterop: requires Linux (dma-buf is Linux-only). "
                   "Found ${CMAKE_SYSTEM_NAME}.")
    return()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(dmabufInterop dmabufInterop.cu)

target_compile_options(dmabufInterop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(dmabufInterop PRIVATE cxx_std_17 cuda_std_17)

target_link_libraries(dmabufInterop PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
)

include(InstallSamples)
setup_samples_install()
