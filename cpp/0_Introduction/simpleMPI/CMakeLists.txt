cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(simpleMPI LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

find_package(MPI)

if(${MPI_FOUND})
    add_executable(simpleMPI simpleMPI.cpp simpleMPI.cu)
    target_compile_options(simpleMPI PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_compile_features(simpleMPI PRIVATE cxx_std_17 cuda_std_17)
    target_include_directories(simpleMPI PUBLIC
        ${MPI_INCLUDE_PATH}
    )
    target_link_libraries(simpleMPI PUBLIC
        ${MPI_C_LIBRARIES}
        ${MPI_CXX_LIBRARIES}
    )
    include(InstallSamples)
    setup_samples_install()
else()
    message(STATUS "MPI not found - will not build sample 'simpleMPI'")
endif()
