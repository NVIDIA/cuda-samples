cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(MonteCarloMultiGPU LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(MonteCarloMultiGPU MonteCarlo_gold.cpp MonteCarlo_kernel.cu multithreading.cpp MonteCarloMultiGPU.cpp)

target_compile_options(MonteCarloMultiGPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(MonteCarloMultiGPU PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(MonteCarloMultiGPU PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(MonteCarloMultiGPU PRIVATE
    CUDA::curand
)

include(InstallSamples)
setup_samples_install()
