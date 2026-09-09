cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(deviceQueryDrv LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(deviceQueryDrv deviceQueryDrv.cpp)

target_compile_options(deviceQueryDrv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(deviceQueryDrv PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(deviceQueryDrv PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(deviceQueryDrv PUBLIC
    CUDA::cuda_driver
)

include(InstallSamples)
setup_samples_install()
