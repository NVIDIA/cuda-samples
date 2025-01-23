cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(nvJPEG LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS "Will not build sample nvJPEG - not supported on aarch64")
else()

    # Source file
    # Add target for nvJPEG
    add_executable(nvJPEG nvJPEG.cpp)

    target_compile_options(nvJPEG PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(nvJPEG PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(nvJPEG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(nvJPEG PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(nvJPEG PRIVATE
        CUDA::cudart
        CUDA::nvjpeg
    )

    # Copy data to the output directory
    add_custom_command(TARGET nvJPEG POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/images
        ${CMAKE_CURRENT_BINARY_DIR}/images
    )

endif()
