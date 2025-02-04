cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleCUDA2GL LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

find_package(OpenGL)
find_package(GLUT)

# Source file
if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            message(STATUS "Will not build sample simpleCUDA2GL - not supported on aarch64")
        else()
            # Add target for simpleCUDA2GL
            add_executable(simpleCUDA2GL simpleCUDA2GL.cu main.cpp)

            target_compile_options(simpleCUDA2GL PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

            target_compile_features(simpleCUDA2GL PRIVATE cxx_std_17 cuda_std_17)

            set_target_properties(simpleCUDA2GL PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

            target_include_directories(simpleCUDA2GL PUBLIC
                ${OPENGL_INCLUDE_DIR}
                ${CUDAToolkit_INCLUDE_DIRS}
                ${GLUT_INCLUDE_DIRS}
            )

        target_link_libraries(simpleCUDA2GL
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )
        endif()
    else()
        message(STATUS "GLUT not found - will not build sample 'simpleCUDA2GL'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'simpleCUDA2GL'")
endif()
