cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(systemWideAtomics LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

include_directories(../../../Common)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS "Will not build sample systemWideAtomics - not supported on aarch64")
else()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(systemWideAtomics systemWideAtomics.cu)

        target_compile_options(systemWideAtomics PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(systemWideAtomics PRIVATE cxx_std_17 cuda_std_17)

    else()
        message(STATUS "Will not build sample systemWideAtomics - requires Linux OS")
    endif()
endif()

include(InstallSamples)
setup_samples_install()
