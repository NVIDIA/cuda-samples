# Include directories and libraries
include_directories(../../../Common)

find_package(EGL)

# Source file
set(SRC_FILES
    cuda_consumer.cpp
    cuda_producer.cpp
    eglstrm_common.cpp
    kernel.cu
    main.cpp
)

if(${EGL_FOUND})
        # Add target for EGLStream_CUDA_CrossGPU
        add_executable(EGLStream_CUDA_CrossGPU ${SRC_FILES})
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
