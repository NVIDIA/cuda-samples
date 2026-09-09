cmake_minimum_required(VERSION 3.20)

set(SAMPLE_DISALLOW_ARCHS 75 80 86 89 90 100 103 107 120)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(nbody_opengles LANGUAGES C CXX CUDA)

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
                add_executable(nbody_opengles bodysystemcuda.cu render_particles.cpp nbody_opengles.cpp)
                target_compile_options(nbody_opengles PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
                target_compile_features(nbody_opengles PRIVATE cxx_std_17 cuda_std_17)
                target_include_directories(nbody_opengles PUBLIC
                    ${EGL_INCLUDE_DIR}
                    ${OPENGL_INCLUDE_DIR}
                    ${CUDAToolkit_INCLUDE_DIRS}
                )
                target_link_libraries(nbody_opengles
                    ${EGL_LIBRARY}
                    ${X11_LIBRARIES}
                    ${OPENGL_LIBRARIES}
                )
                include(InstallSamples)
                setup_samples_install()
            else()
                message(STATUS "X11 libraries not found - will not build sample 'nbody_opengles'")
            endif()
        else()
            message(STATUS "EGL not found - will not build sample 'nbody_opengles'")
        endif()
    else()
        message(STATUS "OpenGL not found - will not build sample 'nbody_opengles'")
    endif()
else()
    message(STATUS "Will not build sample nbody_opengles - requires Linux OS")
endif()
