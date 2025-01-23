cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(oceanFFT LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90 100 120)
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
        # Add target for oceanFFT
        add_executable(oceanFFT oceanFFT.cpp oceanFFT_kernel.cu)

target_compile_options(oceanFFT PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(oceanFFT PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(oceanFFT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(oceanFFT PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(oceanFFT
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
            CUDA::cufft
        )

        # Copy clock_kernel.cu to the output directory
        add_custom_command(TARGET oceanFFT POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
        )

    else()
        message(STATUS "GLUT not found - will not build sample 'oceanFFT'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'oceanFFT'")
endif()
