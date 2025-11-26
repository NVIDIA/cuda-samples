cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/Modules")

project(nbody_opengles LANGUAGES C CXX CUDA)

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
                # Add target for nbody_opengles
                add_executable(nbody_opengles bodysystemcuda.cu render_particles.cpp nbody_opengles.cpp)

                target_compile_options(nbody_opengles PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

                target_compile_features(nbody_opengles PRIVATE cxx_std_17 cuda_std_17)

                set_target_properties(nbody_opengles PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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
