cmake_minimum_required(VERSION 3.20)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(localityDomains LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Source file
# Add target for localityDomains
add_executable(localityDomains localityDomains.cu)

target_compile_options(localityDomains PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(localityDomains PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(localityDomains PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include installation configuration
include(InstallSamples)
setup_samples_install()
