cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(histEqualizationNPP LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
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
    # Add target for histEqualizationNPP
    add_executable(histEqualizationNPP histEqualizationNPP.cpp)

target_compile_options(histEqualizationNPP PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(histEqualizationNPP PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(histEqualizationNPP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(histEqualizationNPP PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${FreeImage_INCLUDE_DIRS}
    )

    target_link_libraries(histEqualizationNPP PRIVATE
        CUDA::nppc
        CUDA::nppisu
        CUDA::nppist
        CUDA::nppicc
        CUDA::cudart
        ${FreeImage_LIBRARIES}
    )
else()
    message(STATUS "FreeImage not found - will not build sample 'histEqualizationNPP'")
endif()
