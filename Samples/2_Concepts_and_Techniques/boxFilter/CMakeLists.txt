cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(boxFilter LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 87 89 90 100 101 120)
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
        # Add target for boxFilter
        add_executable(boxFilter boxFilter.cpp boxFilter_cpu.cpp boxFilter_kernel.cu)

        target_compile_options(boxFilter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(boxFilter PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(boxFilter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(boxFilter PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(boxFilter
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )

        # Copy data files to output directory
        add_custom_command(TARGET boxFilter POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data/teapot1024.ppm
            ${CMAKE_CURRENT_BINARY_DIR}/
        )

        # Copy data files to output directory
        add_custom_command(TARGET boxFilter POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_14.ppm
            ${CMAKE_CURRENT_BINARY_DIR}/
        )

        # Copy data files to output directory
        add_custom_command(TARGET boxFilter POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_22.ppm
            ${CMAKE_CURRENT_BINARY_DIR}/
        )

    else()
        message(STATUS "GLUT not found - will not build sample 'boxFilter'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'boxFilter'")
endif()
