cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(cudaOpenMP LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    add_executable(cudaOpenMP cudaOpenMP.cu)

    target_compile_options(cudaOpenMP PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(cudaOpenMP PRIVATE cxx_std_17 cuda_std_17)

    target_link_libraries(cudaOpenMP PUBLIC
        OpenMP::OpenMP_CXX
    )

    include(InstallSamples)
    setup_samples_install()
else()
    message(STATUS "OpenMP not found - will not build sample 'cudaOpenMP'")
endif()
