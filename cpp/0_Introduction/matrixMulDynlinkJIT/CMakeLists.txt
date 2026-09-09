cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(matrixMulDynlinkJIT LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(matrixMulDynlinkJIT cuda_drvapi_dynlink.c matrixMulDynlinkJIT.cpp matrixMul_gold.cpp matrixMul_kernel_32_ptxdump.c matrixMul_kernel_64_ptxdump.c)

target_compile_options(matrixMulDynlinkJIT PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(matrixMulDynlinkJIT PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(matrixMulDynlinkJIT PROPERTIES
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

include(InstallSamples)
setup_samples_install()
