cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(matrixMulDynlinkJIT LANGUAGES C CXX CUDA)

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

# Source file
# Add target for matrixMulDynlinkJIT
add_executable(matrixMulDynlinkJIT cuda_drvapi_dynlink.c matrixMulDynlinkJIT.cpp matrixMul_gold.cpp matrixMul_kernel_32_ptxdump.c matrixMul_kernel_64_ptxdump.c)

target_compile_options(matrixMulDynlinkJIT PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(matrixMulDynlinkJIT PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(matrixMulDynlinkJIT PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
)

# Only add -no-pie for GCC or Clang
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
endif()

target_link_libraries(matrixMulDynlinkJIT PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    target_link_libraries(matrixMulDynlinkJIT PUBLIC dl)
endif()
