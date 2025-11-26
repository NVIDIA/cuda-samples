cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(streamOrderedAllocationIPC LANGUAGES C CXX CUDA)

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

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
# Source file
# Add target for streamOrderedAllocationIPC
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        message(STATUS "Will not build sample streamOrderedAllocationIPC - not supported on aarch64")
    else()
        add_executable(streamOrderedAllocationIPC streamOrderedAllocationIPC.cu ../../../Common/helper_multiprocess.cpp)

        target_compile_options(streamOrderedAllocationIPC PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(streamOrderedAllocationIPC PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(streamOrderedAllocationIPC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_link_libraries(streamOrderedAllocationIPC PUBLIC
            CUDA::cuda_driver
        )
    endif()
else()
    message(STATUS "Will not build sample streamOrderedAllocationIPC - requires Linux OS")
endif()
