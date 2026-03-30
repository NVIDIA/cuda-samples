cmake_minimum_required(VERSION 3.20)

# Work around a CUDA/glibc declaration conflict (rsqrt/rsqrtf) seen with
# newer glibc versions while keeping standard libc feature declarations.
if(NOT CMAKE_CUDA_FLAGS_INIT MATCHES "(^| )-U_GNU_SOURCE( |$)")
    string(APPEND CMAKE_CUDA_FLAGS_INIT " -U_GNU_SOURCE")
endif()
if(NOT CMAKE_CUDA_FLAGS_INIT MATCHES "(^| )-D_DEFAULT_SOURCE( |$)")
    string(APPEND CMAKE_CUDA_FLAGS_INIT " -D_DEFAULT_SOURCE")
endif()

project(cuda-samples LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets -U_GNU_SOURCE -D_DEFAULT_SOURCE")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -include ${CMAKE_SOURCE_DIR}/cmake/pthread_clock_compat.h")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Add MSVC-specific flags for standard-conforming preprocessor (required for CCCL)
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

# Include installation configuration before processing samples
include(cmake/InstallSamples.cmake)

add_subdirectory(Samples)
