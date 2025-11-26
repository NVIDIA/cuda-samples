cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleD3D11Texture LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

if(WIN32)
    # Source file
    # Add target for simpleD3D11Texture
    add_executable(simpleD3D11Texture
        simpleD3D11Texture.cpp
        ../../../Common/rendercheck_d3d11.cpp
        texture_2d.cu
        texture_3d.cu
        texture_cube.cu
    )

    target_compile_options(simpleD3D11Texture PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(simpleD3D11Texture PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(simpleD3D11Texture PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(simpleD3D11Texture PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(simpleD3D11Texture PRIVATE
        d3d11
        dxgi
        dxguid
        d3dcompiler
    )

    add_custom_command(TARGET simpleD3D11Texture POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        ${CMAKE_CURRENT_BINARY_DIR}/data
    )
else()
    message(STATUS "Sample 'simpleD3D11Texture' is Windows-only - skipping")
endif()
