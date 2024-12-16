cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(matrixMulDynlinkJIT LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "native")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    cuda_drvapi_dynlink.c
    matrixMulDynlinkJIT.cpp
    matrixMul_gold.cpp
    matrixMul_kernel_32_ptxdump.c
    matrixMul_kernel_64_ptxdump.c
)



# Add target for matrixMulDynlinkJIT
add_executable(matrixMulDynlinkJIT ${SRC_FILES})
set_target_properties(matrixMulDynlinkJIT PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
target_link_libraries(matrixMulDynlinkJIT PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)
