cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(cuDLAHybridMode LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 53 61 70 72 75 80 86 87 90)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../../Common)

find_library(CUDLA_LIB cudla PATHS ${CUDAToolkit_LIBRARY_DIR})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CUDLA_LIB)
        # Source file
        # Add target for cuDLAHybridMode
        add_executable(cuDLAHybridMode main.cu)

        target_compile_options(cuDLAHybridMode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(cuDLAHybridMode PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(cuDLAHybridMode PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(cuDLAHybridMode PUBLIC
            ${CUDAToolkit_INCLUDE_DIRS}
        )

        target_link_libraries(cuDLAHybridMode
            ${CUDLA_LIB}
        )
    else()
        message(STATUS "CUDLA not found - will not build sample 'cuDLAHybridMode'")
    endif()
else()
    message(STATUS "Will not build sample cuDLAHybridMode - requires Linux OS")
endif()
