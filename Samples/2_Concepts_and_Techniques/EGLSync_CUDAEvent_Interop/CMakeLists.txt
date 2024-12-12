# Include directories and libraries
include_directories(../../../Common)

find_package(EGL)
find_package(X11)
find_package(OpenGL)

# Source file
set(SRC_FILES
    EGLSync_CUDAEvent_Interop.cu
)

if(${OpenGL_FOUND})
    if(${EGL_FOUND})
        if(${X11_FOUND})
            # Add target for EGLSync_CUDAEvent_Interop
            add_executable(EGLSync_CUDAEvent_Interop ${SRC_FILES})
            set_target_properties(EGLSync_CUDAEvent_Interop PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

            target_include_directories(EGLSync_CUDAEvent_Interop PUBLIC
                ${CUDAToolkit_INCLUDE_DIRS}
                ${EGL_INCLUDE_DIR}
                ${X11_INCLUDE_DIR}
                ${OPENGL_INCLUDE_DIR}
            )

            target_link_libraries(EGLSync_CUDAEvent_Interop
                CUDA::cuda_driver
                #CUDA::cudart
                ${EGL_LIBRARY}
                ${X11_LIBRARIES}
                ${OPENGL_LIBRARIES}
            )
        else()
            message(STATUS "X11 libraries not found - will not build sample 'EGLSyncCUDAEvent_Interop'")
        endif()
    else()
        message(STATUS "EGL not found - will not build sample 'EGLSync_CUDAEvent_Interop'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'EGLSync_CUDAEvent_Interop'")
endif()
