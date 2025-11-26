cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(EGLStream_CUDA_CrossGPU LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

find_package(EGL)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Source file
    if(${EGL_FOUND})
            # Add target for EGLStream_CUDA_CrossGPU
            add_executable(EGLStream_CUDA_CrossGPU cuda_consumer.cpp cuda_producer.cpp eglstrm_common.cpp kernel.cu main.cpp)

            target_compile_options(EGLStream_CUDA_CrossGPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

            target_compile_features(EGLStream_CUDA_CrossGPU PRIVATE cxx_std_17 cuda_std_17)

            set_target_properties(EGLStream_CUDA_CrossGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

            target_include_directories(EGLStream_CUDA_CrossGPU PUBLIC
                ${EGL_INCLUDE_DIR}
                ${CUDAToolkit_INCLUDE_DIRS}
            )

            target_link_libraries(EGLStream_CUDA_CrossGPU
                ${EGL_LIBRARY}
                CUDA::cuda_driver
            )
    else()
        message(STATUS "EGL not found - will not build sample 'EGLStream_CUDA_CrossGPU'")
    endif()
else()
    message(STATUS "Will not build sample EGLStream_CUDA_CrossGPU - requires Linux OS")
endif()
