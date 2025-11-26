cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/Modules")

project(cuDLALayerwiseStatsHybrid LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 87 110)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../../Common)

find_library(CUDLA_LIB cudla PATHS ${CUDAToolkit_LIBRARY_DIR} ${CMAKE_LIBRARY_PATH})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CUDLA_LIB)
        # Source file
        # Add target for cuDLALayerwiseStatsHybrid
        add_executable(cuDLALayerwiseStatsHybrid main.cu)

        target_compile_options(cuDLALayerwiseStatsHybrid PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(cuDLALayerwiseStatsHybrid PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(cuDLALayerwiseStatsHybrid PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(cuDLALayerwiseStatsHybrid PUBLIC
            ${CUDAToolkit_INCLUDE_DIRS}
        )

        target_link_libraries(cuDLALayerwiseStatsHybrid
            ${CUDLA_LIB}
        )
    else()
        message(STATUS "CUDLA not found - will not build sample 'cuDLALayerwiseStatsHybrid'")
    endif()
else()
    message(STATUS "Will not build sample cuDLALayerwiseStatsHybrid - requires Linux OS")
endif()
