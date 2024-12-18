cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(simpleD3D12 LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

if(WIN32)
    find_package(DirectX)

    if(${DirectX_FOUND})

        # Source file
        # Add target for simpleD3D12
        add_executable(simpleD3D12
            DX12CudaSample.cpp
            Main.cpp
            Win32Application.cpp
            simpleD3D12.cpp
            stdafx.cpp
            sinewave_cuda.cu
        )

        target_compile_options(simpleD3D12 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(simpleD3D12 PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(simpleD3D12 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_link_libraries(simpleD3D12 PRIVATE
            DirectX::D3D12
        )

        add_custom_command(TARGET simpleD3D11 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_file
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders.hlsl
            ${CMAKE_CURRENT_BINARY_DIR}/shaders.hlsl
        )

    else()
        message(STATUS "DirectX not found - will not build sample 'simpleD3D12'")
    endif()
endif()
