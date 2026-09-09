cmake_minimum_required(VERSION 3.20)

set(SAMPLE_DISALLOW_ARCHS 75 80 86 89 90 100 103 107 120)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(simpleGLES_EGLOutput LANGUAGES C CXX CUDA)

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
                set(DRM_INCLUDE_PATH "/usr/include/drm" "/usr/include/libdrm" ${CMAKE_INCLUDE_PATH})

                # use CMAKE_LIBRARY_PATH so that users can also specify the NVSCI lib path in cmake command
                set(CMAKE_LIBRARY_PATH "/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu" ${CMAKE_LIBRARY_PATH})
                find_library(DRM_LIB drm PATHS ${CMAKE_LIBRARY_PATH})

                add_executable(simpleGLES_EGLOutput simpleGLES_EGLOutput.cu)
                target_compile_options(simpleGLES_EGLOutput PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
                target_compile_features(simpleGLES_EGLOutput PRIVATE cxx_std_17 cuda_std_17)
                target_include_directories(simpleGLES_EGLOutput PUBLIC
                    ${EGL_INCLUDE_DIR}
                    ${OPENGL_INCLUDE_DIR}
                    ${CUDAToolkit_INCLUDE_DIR}
                    ${DRM_INCLUDE_PATH}
                )
                target_link_libraries(simpleGLES_EGLOutput
                    ${EGL_LIBRARY}
                    ${X11_LIBRARIES}
                    ${OPENGL_LIBRARIES}
                    ${DRM_LIB}
                )
                # Copy the .glsl files to the output directory
                add_custom_command(TARGET simpleGLES_EGLOutput POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${CMAKE_CURRENT_SOURCE_DIR}/mesh.frag.glsl
                    ${CMAKE_CURRENT_SOURCE_DIR}/mesh.vert.glsl
                    ${CMAKE_CURRENT_BINARY_DIR}
                )
                include(InstallSamples)
                setup_samples_install()
            else()
                message(STATUS "X11 libraries not found - will not build sample 'simpleGLES_EGLOutput'")
            endif()
        else()
            message(STATUS "EGL not found - will not build sample 'simpleGLES_EGLOutput'")
        endif()
    else()
        message(STATUS "OpenGL not found - will not build sample 'simpleGLES_EGLOutput'")
    endif()
else()
    message(STATUS "Will not build sample simpleGLES_EGLOutput - requires Linux OS")
endif()
