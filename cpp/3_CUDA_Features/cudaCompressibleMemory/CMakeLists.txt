cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(cudaCompressibleMemory LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(cudaCompressibleMemory compMalloc.cpp saxpy.cu)

target_compile_options(cudaCompressibleMemory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cudaCompressibleMemory PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(cudaCompressibleMemory PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cudaCompressibleMemory PRIVATE
    CUDA::cuda_driver
)

include(InstallSamples)
setup_samples_install()
