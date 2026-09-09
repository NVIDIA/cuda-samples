cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(FDTD3d LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(FDTD3d src/FDTD3d.cpp src/FDTD3dGPU.cu src/FDTD3dReference.cpp)

target_compile_options(FDTD3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(FDTD3d PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(FDTD3d PRIVATE
    inc
)

include(InstallSamples)
setup_samples_install()
