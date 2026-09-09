cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(UnifiedMemoryStreams LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

# This sample is not supported on QNX
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    message(STATUS "Will not build sample ${PROJECT_NAME} - not supported on QNX")
    return()
endif()

# FindOpenMP: request COMPONENTS CXX only. From CMake 3.31 on, a bare find_package(OpenMP) also
# probes OpenMP for CUDA when the project enables CUDA; that check can fail even when C++ OpenMP
# works, and this sample only needs the C++ OpenMP package. Link OpenMP::OpenMP_CXX for libs/headers.
find_package(OpenMP COMPONENTS CXX)

if(OpenMP_CXX_FOUND)
    add_executable(UnifiedMemoryStreams UnifiedMemoryStreams.cu)
    target_compile_options(UnifiedMemoryStreams PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_compile_features(UnifiedMemoryStreams PRIVATE cxx_std_17 cuda_std_17)
    target_link_libraries(UnifiedMemoryStreams PUBLIC
        CUDA::cublas
        OpenMP::OpenMP_CXX
    )
    include(InstallSamples)
    setup_samples_install()
else()
    message(STATUS "OpenMP not found - will not build sample 'UnifiedMemoryStreams'")
endif()
