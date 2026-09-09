cmake_minimum_required(VERSION 3.20)

set(SAMPLE_DISALLOW_ARCHS 75 80 86 89 90 100 103 107 120)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(EGLSync_CUDAEvent_Interop LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../../Common)

find_package(EGL)
find_package(X11)
find_package(OpenGL)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(${OpenGL_FOUND})
        if(${EGL_FOUND})
            if(${X11_FOUND})
                add_executable(EGLSync_CUDAEvent_Interop EGLSync_CUDAEvent_Interop.cu)
                target_compile_options(EGLSync_CUDAEvent_Interop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
                target_compile_features(EGLSync_CUDAEvent_Interop PRIVATE cxx_std_17 cuda_std_17)
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
                include(InstallSamples)
                setup_samples_install()
            else()
                message(STATUS "X11 libraries not found - will not build sample 'EGLSync_CUDAEvent_Interop'")
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
