cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(freeImageInteropNPP LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(
    ../../../Common
    ../../../Common/UtilNPP
)

# Source file
find_package(FreeImage)

if(${FreeImage_FOUND})
    # Add target for freeImageInteropNPP
    add_executable(freeImageInteropNPP freeImageInteropNPP.cpp)

target_compile_options(freeImageInteropNPP PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(freeImageInteropNPP PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(freeImageInteropNPP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(freeImageInteropNPP PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${FreeImage_INCLUDE_DIRS}
    )

    target_link_libraries(freeImageInteropNPP PRIVATE
        CUDA::nppc
        CUDA::nppisu
        CUDA::nppif
        CUDA::cudart
        ${FreeImage_LIBRARIES}
    )
else()
    message(STATUS "FreeImage not found - will not build sample 'freeImageInteropNPP'")
endif()

# Copy data files to output directory
add_custom_command(TARGET freeImageInteropNPP POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../Common/data/teapot512.pgm
    ${CMAKE_CURRENT_BINARY_DIR}/
)
