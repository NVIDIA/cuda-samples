cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(vectorAdd_nvrtc LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

# Add sample target executable
add_executable(vectorAdd_nvrtc vectorAdd.cpp)

target_compile_options(vectorAdd_nvrtc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(vectorAdd_nvrtc PRIVATE cxx_std_17 cuda_std_17)

target_link_libraries(vectorAdd_nvrtc PRIVATE
    CUDA::nvrtc
    CUDA::cuda_driver
)

# Copy vectorAdd_kernel.cu to the output directory
add_custom_command(TARGET vectorAdd_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/vectorAdd_kernel.cu ${CMAKE_CURRENT_BINARY_DIR}
)

include(InstallSamples)
setup_samples_install()
