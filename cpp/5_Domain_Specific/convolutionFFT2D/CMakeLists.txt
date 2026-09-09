cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(convolutionFFT2D LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(convolutionFFT2D convolutionFFT2D.cu convolutionFFT2D_gold.cpp main.cpp)

target_compile_options(convolutionFFT2D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(convolutionFFT2D PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(convolutionFFT2D PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(convolutionFFT2D PUBLIC
    CUDA::cufft
)

include(InstallSamples)
setup_samples_install()
