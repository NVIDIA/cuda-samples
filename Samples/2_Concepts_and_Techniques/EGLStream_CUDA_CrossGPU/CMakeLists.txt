cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(EGLStream_CUDA_CrossGPU LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
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
