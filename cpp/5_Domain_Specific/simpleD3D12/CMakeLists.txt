cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(simpleD3D12 LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

if(WIN32)
    add_executable(simpleD3D12 WIN32
        DX12CudaSample.cpp
        Main.cpp
        Win32Application.cpp
        simpleD3D12.cpp
        stdafx.cpp
        sinewave_cuda.cu
    )
    target_compile_options(simpleD3D12 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_compile_features(simpleD3D12 PRIVATE cxx_std_17 cuda_std_17)
    target_include_directories(simpleD3D12 PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(simpleD3D12 PRIVATE
        d3d12
        dxgi
        dxguid
        d3dcompiler
    )
    add_custom_command(TARGET simpleD3D12 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders.hlsl
        ${CMAKE_CURRENT_BINARY_DIR}/shaders.hlsl
    )
    include(InstallSamples)
    setup_samples_install()
else()
    message(STATUS "Sample 'simpleD3D12' is Windows-only - skipping")
endif()
