cmake_minimum_required(VERSION 3.20)

set(SAMPLE_DISALLOW_ARCHS 75 80 86 89 90 100 103 107 120)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(cuDLAHybridMode LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../../Common)

find_library(CUDLA_LIB cudla PATHS ${CUDAToolkit_LIBRARY_DIR} ${CMAKE_LIBRARY_PATH})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CUDLA_LIB)
        add_executable(cuDLAHybridMode main.cu)
        target_compile_options(cuDLAHybridMode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        target_compile_features(cuDLAHybridMode PRIVATE cxx_std_17 cuda_std_17)
        target_include_directories(cuDLAHybridMode PUBLIC
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_link_libraries(cuDLAHybridMode
            ${CUDLA_LIB}
        )
        include(InstallSamples)
        setup_samples_install()
    else()
        message(STATUS "CUDLA not found - will not build sample 'cuDLAHybridMode'")
    endif()
else()
    message(STATUS "Will not build sample cuDLAHybridMode - requires Linux OS")
endif()
