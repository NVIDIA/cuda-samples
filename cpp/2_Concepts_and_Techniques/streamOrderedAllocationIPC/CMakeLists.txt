cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(streamOrderedAllocationIPC LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        message(STATUS "Will not build sample streamOrderedAllocationIPC - not supported on aarch64")
    else()
        add_executable(streamOrderedAllocationIPC streamOrderedAllocationIPC.cu ../../../Common/helper_multiprocess.cpp)

        target_compile_options(streamOrderedAllocationIPC PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(streamOrderedAllocationIPC PRIVATE cxx_std_17 cuda_std_17)

        target_link_libraries(streamOrderedAllocationIPC PUBLIC
            CUDA::cuda_driver
        )
    endif()
else()
    message(STATUS "Will not build sample streamOrderedAllocationIPC - requires Linux OS")
endif()

include(InstallSamples)
setup_samples_install()
