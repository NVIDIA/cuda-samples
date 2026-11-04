cmake_minimum_required(VERSION 3.20)

# The shared-context integration test builds for SM 8.7 and later.
set(SAMPLE_DISALLOW_ARCHS 75 80 86)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(sharedCtx LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

# cudaSafeExQueryDeviceKey / Export / Import ship only in the CUDA Safe runtime.
set(_safe_ex_header "")
foreach(_inc ${CUDAToolkit_INCLUDE_DIRS})
    if(EXISTS "${_inc}/cuda_runtime_api_safe_ex.h")
        set(_safe_ex_header "${_inc}/cuda_runtime_api_safe_ex.h")
    endif()
endforeach()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "QNX" OR _safe_ex_header STREQUAL "")
    message(STATUS "Will not build sample sharedCtx - requires the CUDA Safe toolkit on QNX")
    return()
endif()

include_directories(../../../Common)

add_executable(sharedCtx sharedCtx.cu)

target_compile_options(sharedCtx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(sharedCtx PRIVATE cxx_std_17 cuda_std_17)

target_link_libraries(sharedCtx PRIVATE socket)

include(InstallSamples)
setup_samples_install()
