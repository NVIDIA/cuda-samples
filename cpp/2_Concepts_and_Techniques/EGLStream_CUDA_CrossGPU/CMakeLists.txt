cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(EGLStream_CUDA_CrossGPU LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

find_package(EGL)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(${EGL_FOUND})
        add_executable(EGLStream_CUDA_CrossGPU cuda_consumer.cpp cuda_producer.cpp eglstrm_common.cpp kernel.cu main.cpp)
        target_compile_options(EGLStream_CUDA_CrossGPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        target_compile_features(EGLStream_CUDA_CrossGPU PRIVATE cxx_std_17 cuda_std_17)
        target_include_directories(EGLStream_CUDA_CrossGPU PUBLIC
            ${EGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_link_libraries(EGLStream_CUDA_CrossGPU
            ${EGL_LIBRARY}
            CUDA::cuda_driver
        )
        include(InstallSamples)
        setup_samples_install()
    else()
        message(STATUS "EGL not found - will not build sample 'EGLStream_CUDA_CrossGPU'")
    endif()
else()
    message(STATUS "Will not build sample EGLStream_CUDA_CrossGPU - requires Linux OS")
endif()
