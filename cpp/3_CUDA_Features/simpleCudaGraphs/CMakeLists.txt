cmake_minimum_required(VERSION 3.20)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(simpleCudaGraphs LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

add_executable(simpleCudaGraphs_explicit simpleCudaGraphs_explicit.cu)
add_executable(simpleCudaGraphs_capture  simpleCudaGraphs_capture.cu)

foreach(target simpleCudaGraphs_explicit simpleCudaGraphs_capture)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_compile_features(${target} PRIVATE cxx_std_17 cuda_std_17)
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${target} PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
endforeach()

include(InstallSamples)
setup_samples_install()
