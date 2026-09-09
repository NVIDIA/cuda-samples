cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(cubDeviceTransform LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

# Disable response file for libraries on QNX as qcc does not support lib paths with double quotes
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
endif()

find_package(CUDAToolkit REQUIRED)

# Fetch CCCL via CPM.
# Override with -DCCCL_SOURCE_DIR=/path/to/cccl to use a local checkout
set(CCCL_SAMPLES_CCCL_TAG "v3.3.3" CACHE STRING
    "Tag/branch of NVIDIA/cccl to fetch for the CCCL samples")

if(NOT TARGET CCCL::CCCL)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CPM.cmake")
    if(DEFINED CCCL_SOURCE_DIR AND NOT CCCL_SOURCE_DIR STREQUAL "")
        CPMAddPackage(NAME CCCL SOURCE_DIR "${CCCL_SOURCE_DIR}")
    else()
        CPMAddPackage(
            NAME CCCL
            GIT_REPOSITORY "https://github.com/NVIDIA/cccl"
            GIT_TAG "${CCCL_SAMPLES_CCCL_TAG}"
        )
    endif()
endif()

include_directories(../../../Common)

add_executable(cubDeviceTransform cubDeviceTransform.cu)

target_compile_options(cubDeviceTransform PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cubDeviceTransform PRIVATE cxx_std_17 cuda_std_17)

target_link_libraries(cubDeviceTransform PRIVATE
    CUDA::cudart
    CCCL::CCCL
)

include(InstallSamples)
setup_samples_install()
