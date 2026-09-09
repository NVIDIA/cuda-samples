cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(HSOpticalFlow LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(HSOpticalFlow flowCUDA.cu flowGold.cpp main.cpp)

target_compile_options(HSOpticalFlow PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(HSOpticalFlow PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(HSOpticalFlow PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

add_custom_command(TARGET HSOpticalFlow POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_BINARY_DIR}/data
)

include(InstallSamples)
setup_samples_install()
