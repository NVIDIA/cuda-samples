cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(nvJPEG_encoder LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS "Will not build sample nvJPEG_encoder - not supported on aarch64")
else()

    # Source file
    # Add target for nvJPEG_encoder
    add_executable(nvJPEG_encoder nvJPEG_encoder.cpp)

    target_compile_options(nvJPEG_encoder PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(nvJPEG_encoder PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(nvJPEG_encoder PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(nvJPEG_encoder PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(nvJPEG_encoder PRIVATE
        CUDA::cudart
        CUDA::nvjpeg
    )

    # Copy data to the output directory
    add_custom_command(TARGET nvJPEG_encoder POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/images
        ${CMAKE_CURRENT_BINARY_DIR}/images
    )

    add_custom_command(TARGET nvJPEG_encoder POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/encode_output
        ${CMAKE_CURRENT_BINARY_DIR}/encode_output
    )

endif()
