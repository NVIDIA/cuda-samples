cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(quasirandomGenerator LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(quasirandomGenerator quasirandomGenerator.cpp quasirandomGenerator_gold.cpp quasirandomGenerator_kernel.cu)

target_compile_options(quasirandomGenerator PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(quasirandomGenerator PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(quasirandomGenerator PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

include(InstallSamples)
setup_samples_install()
