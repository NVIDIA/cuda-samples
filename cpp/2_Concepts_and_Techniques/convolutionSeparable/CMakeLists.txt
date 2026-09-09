cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(convolutionSeparable LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(convolutionSeparable convolutionSeparable.cu convolutionSeparable_gold.cpp main.cpp)

target_compile_options(convolutionSeparable PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(convolutionSeparable PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(convolutionSeparable PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)

include(InstallSamples)
setup_samples_install()
