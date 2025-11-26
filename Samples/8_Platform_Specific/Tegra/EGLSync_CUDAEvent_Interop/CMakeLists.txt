cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/Modules")

project(EGLSync_CUDAEvent_Interop LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 87 110)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../../Common)

find_package(EGL)
find_package(X11)
find_package(OpenGL)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Source file
    if(${OpenGL_FOUND})
        if(${EGL_FOUND})
            if(${X11_FOUND})
                # Add target for EGLSync_CUDAEvent_Interop
                add_executable(EGLSync_CUDAEvent_Interop EGLSync_CUDAEvent_Interop.cu)

                target_compile_options(EGLSync_CUDAEvent_Interop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

                target_compile_features(EGLSync_CUDAEvent_Interop PRIVATE cxx_std_17 cuda_std_17)

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
else()
    message(STATUS "Will not build sample EGLSync_CUDAEvent_Interop
 - requires Linux OS")
endif()
