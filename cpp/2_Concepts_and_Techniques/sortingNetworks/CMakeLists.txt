cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(sortingNetworks LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(sortingNetworks main.cpp bitonicSort.cu sortingNetworks_validate.cpp)

target_compile_options(sortingNetworks PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(sortingNetworks PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(sortingNetworks PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)

include(InstallSamples)
setup_samples_install()
