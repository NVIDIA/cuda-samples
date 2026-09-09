cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(MC_EstimatePiInlineP LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(MC_EstimatePiInlineP src/main.cpp src/piestimator.cu src/test.cpp)

target_compile_options(MC_EstimatePiInlineP PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(MC_EstimatePiInlineP PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(MC_EstimatePiInlineP PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CUDAToolkit_INCLUDE_DIRS}
)

include(InstallSamples)
setup_samples_install()
