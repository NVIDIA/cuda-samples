cmake_minimum_required(VERSION 3.20)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(localityDomainsDrv LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Source file
# Add target for localityDomainsDrv
add_executable(localityDomainsDrv localityDomainsDrv.cpp)

target_compile_options(localityDomainsDrv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(localityDomainsDrv PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(localityDomainsDrv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(localityDomainsDrv PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(localityDomainsDrv PUBLIC
    CUDA::cuda_driver
)

# Include installation configuration
include(InstallSamples)
setup_samples_install()
