cmake_minimum_required(VERSION 3.20)

# CDP device-links the CUDA device runtime (cudadevrt), which the DriveOS Tegra
# toolkit ships only for the native Tegra archs; drop the desktop archs there.
get_filename_component(_ctk_root "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(_ctk_root "${_ctk_root}" DIRECTORY)
if(EXISTS "${_ctk_root}/targets/aarch64-linux" OR EXISTS "${_ctk_root}/targets/aarch64-qnx")
    set(SAMPLE_DISALLOW_ARCHS 75 80 86 89 90 100 103 107 120)
endif()
unset(_ctk_root)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(cdpSimplePrint LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(cdpSimplePrint cdpSimplePrint.cu)

target_compile_options(cdpSimplePrint PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cdpSimplePrint PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(cdpSimplePrint PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include(InstallSamples)
setup_samples_install()
