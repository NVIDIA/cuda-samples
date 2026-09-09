cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(vulkanImageCUDA LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

find_package(Vulkan)
find_package(OpenGL)

# Include the check_include_file macro
include(CheckIncludeFile)

# Check for the GLFW/glfw3.h header
check_include_file("GLFW/glfw3.h" HAVE_GLFW3_H)

# Check for SUSE Linux
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    file(READ "/etc/os-release" OS_RELEASE_CONTENTS)
    if(OS_RELEASE_CONTENTS MATCHES "ID=[^ ]*suse[^ ]*")
        message(STATUS "SUSE Linux detected")
        set(SUSE_LINUX TRUE)
    endif()
endif()

# Find GLFW header and lib for Windows and SUSE Linux
if(WIN32 OR CMAKE_CROSSCOMPILING OR SUSE_LINUX)
    set(GLFW_INCLUDE_DIRS "${GLFW_INCLUDE_DIR}" "${CMAKE_INCLUDE_PATH}")
    find_file(GLFW3_H "GLFW/glfw3.h" PATHS ${GLFW_INCLUDE_DIRS})
    find_library(GLFW3_LIB
                 NAMES glfw3 glfw
                 PATHS "${GLFW_LIB_DIR}"
                 PATH_SUFFIXES lib lib64)
    if(GLFW3_H AND GLFW3_LIB)
        message(STATUS "Found GLFW/glfw3.h and GLFW library.")
        set(HAVE_GLFW3_H 1)
    endif()
endif()

if(${Vulkan_FOUND})
    if(${OPENGL_FOUND})
        if(${HAVE_GLFW3_H})
            add_executable(vulkanImageCUDA vulkanImageCUDA.cu)
            target_compile_options(vulkanImageCUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
            target_compile_features(vulkanImageCUDA PRIVATE cxx_std_17 cuda_std_17)
            target_include_directories(vulkanImageCUDA PUBLIC
                ${Vulkan_INCLUDE_DIRS}
                ${CUDAToolkit_INCLUDE_DIRS}
            )
            target_link_libraries(vulkanImageCUDA
                ${Vulkan_LIBRARIES}
                OpenGL::GL
            )
            if(WIN32 OR CMAKE_CROSSCOMPILING OR SUSE_LINUX)
                target_include_directories(vulkanImageCUDA PUBLIC
                    ${GLFW_INCLUDE_DIRS}
                )
                target_link_libraries(vulkanImageCUDA
                    ${GLFW3_LIB}
                )
            else()
                target_link_libraries(vulkanImageCUDA
                    glfw
                )
            endif()
            # Need to add X11 for SUSE Linux explicitly
            if(SUSE_LINUX)
                target_link_libraries(vulkanImageCUDA
                    X11
                )
            endif()
            add_custom_command(TARGET vulkanImageCUDA POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag
                ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert
                ${CMAKE_CURRENT_SOURCE_DIR}/vert.spv
                ${CMAKE_CURRENT_SOURCE_DIR}/frag.spv
                ${CMAKE_CURRENT_SOURCE_DIR}/teapot1024.ppm
                ${CMAKE_CURRENT_BINARY_DIR}
            )
            include(InstallSamples)
            setup_samples_install()
        else()
            message(STATUS "glfw3 not found - will not build sample 'vulkanImageCUDA'")
        endif()
    else()
        message(STATUS "OpenGL not found - will not build sample 'vulkanImageCUDA'")
    endif()
else()
    message(STATUS "Vulkan not found - will not build sample 'vulkanImageCUDA'")
endif()
