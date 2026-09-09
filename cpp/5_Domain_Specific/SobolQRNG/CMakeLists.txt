cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(SobolQRNG LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(SobolQRNG sobol_gold.cpp sobol_gpu.cu sobol_primitives.cpp sobol.cpp)

target_compile_options(SobolQRNG PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(SobolQRNG PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(SobolQRNG PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

include(InstallSamples)
setup_samples_install()
