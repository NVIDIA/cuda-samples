cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(simpleCallback LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(simpleCallback simpleCallback.cu)

target_compile_options(simpleCallback PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleCallback PRIVATE cxx_std_17 cuda_std_17)

# std::thread is backed by pthreads on Linux
find_package(Threads REQUIRED)
target_link_libraries(simpleCallback PRIVATE Threads::Threads)


include(InstallSamples)
setup_samples_install()
