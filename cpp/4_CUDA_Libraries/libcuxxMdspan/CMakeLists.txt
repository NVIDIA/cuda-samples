cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(libcuxxMdspan LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

# Disable response file for libraries on QNX as qcc does not support lib paths with double quotes
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
endif()

find_package(CUDAToolkit REQUIRED)

# Fetch CCCL (CUB + libcu++ + Thrust) via CPM. The toolkit that ships
# with CUDA 13.2 bundles CCCL 3.2, but this sample uses APIs added in
# CCCL 3.3. Pinning the tag here lets the sample build on any toolkit
# with a usable nvcc. Override with -DCCCL_SOURCE_DIR=/path/to/cccl
# to use a local checkout instead of fetching from GitHub.
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

# DLPack headers are required for cuda::to_device_mdspan and
# cuda::to_dlpack_tensor. Fetch via CPM (override with DLPACK_SOURCE_DIR).
set(CCCL_SAMPLES_DLPACK_TAG "v1.3" CACHE STRING
    "Tag of dmlc/dlpack to fetch for libcuxxMdspan")

if(NOT TARGET dlpack::dlpack)
    if(NOT COMMAND CPMAddPackage)
        include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CPM.cmake")
    endif()
    if(DEFINED DLPACK_SOURCE_DIR AND NOT DLPACK_SOURCE_DIR STREQUAL "")
        CPMAddPackage(NAME dlpack SOURCE_DIR "${DLPACK_SOURCE_DIR}" DOWNLOAD_ONLY YES)
    else()
        CPMAddPackage(
            NAME dlpack
            GIT_REPOSITORY "https://github.com/dmlc/dlpack"
            GIT_TAG "${CCCL_SAMPLES_DLPACK_TAG}"
            DOWNLOAD_ONLY YES
        )
    endif()
    if(dlpack_ADDED)
        add_library(dlpack_headers INTERFACE)
        target_include_directories(dlpack_headers INTERFACE "${dlpack_SOURCE_DIR}/include")
        add_library(dlpack::dlpack ALIAS dlpack_headers)
    endif()
endif()

include_directories(../../../Common)

add_executable(libcuxxMdspan libcuxxMdspan.cu)

target_compile_options(libcuxxMdspan PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(libcuxxMdspan PRIVATE cxx_std_17 cuda_std_17)

target_link_libraries(libcuxxMdspan PRIVATE
    CUDA::cudart
    CCCL::CCCL
    dlpack::dlpack
)

include(InstallSamples)
setup_samples_install()
