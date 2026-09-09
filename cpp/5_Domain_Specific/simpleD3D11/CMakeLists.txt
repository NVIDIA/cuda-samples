cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(simpleD3D11 LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

if(WIN32)
    add_executable(simpleD3D11
        simpleD3D11.cpp
        sinewave_cuda.cu
        ../../../Common/rendercheck_d3d11.cpp
    )
    target_compile_options(simpleD3D11 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_compile_features(simpleD3D11 PRIVATE cxx_std_17 cuda_std_17)
    target_include_directories(simpleD3D11 PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(simpleD3D11 PRIVATE
        d3d11
        dxgi
        dxguid
        d3dcompiler
    )
    add_custom_command(TARGET simpleD3D11 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        ${CMAKE_CURRENT_BINARY_DIR}/data
    )
    include(InstallSamples)
    setup_samples_install()
else()
    message(STATUS "Sample 'simpleD3D11' is Windows-only - skipping")
endif()
