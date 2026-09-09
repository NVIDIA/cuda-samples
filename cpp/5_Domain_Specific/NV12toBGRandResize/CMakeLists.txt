cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(NV12toBGRandResize LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(NV12toBGRandResize bgr_resize.cu nv12_resize.cu nv12_to_bgr_planar.cu resize_convert_main.cpp utils.cu)

target_compile_options(NV12toBGRandResize PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(NV12toBGRandResize PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(NV12toBGRandResize PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

add_custom_command(TARGET NV12toBGRandResize POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_BINARY_DIR}/data
)

include(InstallSamples)
setup_samples_install()
