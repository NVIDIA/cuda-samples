cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(volumeFiltering LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

if(WIN32)
    set(PC_GLUT_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../Common")
    set(PC_GLUT_LIBRARY_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../Common/lib/x64")
endif()

find_package(OpenGL)
find_package(GLUT)

# Source file
if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        # Add target for volumeFiltering
        add_executable(volumeFiltering volume.cpp volumeFilter_kernel.cu volumeFiltering.cpp volumeRender_kernel.cu)

        target_compile_options(volumeFiltering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(volumeFiltering PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(volumeFiltering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(volumeFiltering PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(volumeFiltering
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )

        add_custom_command(TARGET volumeFiltering POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
        )

        if(WIN32)
            target_link_libraries(volumeFiltering
                ${PC_GLUT_LIBRARY_DIRS}/freeglut.lib
                ${PC_GLUT_LIBRARY_DIRS}/glew64.lib
            )

            add_custom_command(TARGET volumeFiltering
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/../../../bin/win64/$<CONFIGURATION>/freeglut.dll
                ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>
            )

            add_custom_command(TARGET volumeFiltering
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/../../../bin/win64/$<CONFIGURATION>/glew64.dll
                ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>
            )
        endif()

    else()
        message(STATUS "GLUT not found - will not build sample 'volumeFiltering'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'volumeFiltering'")
endif()
