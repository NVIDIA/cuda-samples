cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleVulkan LANGUAGES C CXX CUDA)

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

find_package(Vulkan)
find_package(OpenGL)


# Include the check_include_file macro
include(CheckIncludeFile)

# Check for the GLFW/glfw3.h header
check_include_file("GLFW/glfw3.h" HAVE_GLFW3_H)

# Find GLFW header and lib for Windows
if(WIN32 OR CMAKE_CROSSCOMPILING)
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

# Source file
if(${Vulkan_FOUND})
    if(${OPENGL_FOUND})
        if(${HAVE_GLFW3_H})
            # Add target for simpleVulkan
            add_executable(simpleVulkan main.cpp SineWaveSimulation.cu VulkanBaseApp.cpp)

            target_compile_options(simpleVulkan PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

            target_compile_features(simpleVulkan PRIVATE cxx_std_17 cuda_std_17)

            set_target_properties(simpleVulkan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

            target_include_directories(simpleVulkan PUBLIC
                ${Vulkan_INCLUDE_DIRS}
                ${CUDAToolkit_INCLUDE_DIRS}
            )
            target_link_libraries(simpleVulkan
                ${Vulkan_LIBRARIES}
                OpenGL::GL
            )
            if(WIN32 OR CMAKE_CROSSCOMPILING)
                target_include_directories(simpleVulkan PUBLIC
                    ${GLFW_INCLUDE_DIRS}
                )
                target_link_libraries(simpleVulkan
                    ${GLFW3_LIB}
                )
            else()
                target_link_libraries(simpleVulkan
                    glfw
                )
            endif()
            add_custom_command(TARGET simpleVulkan POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/sinewave.frag
                ${CMAKE_CURRENT_SOURCE_DIR}/sinewave.vert
                ${CMAKE_CURRENT_SOURCE_DIR}/vert.spv
                ${CMAKE_CURRENT_SOURCE_DIR}/frag.spv
                ${CMAKE_CURRENT_BINARY_DIR}
            )
        else()
            message(STATUS "glfw3 not found - will not build sample 'simpleVulkan'")
        endif()
    else()
        message(STATUS "OpenGL not found - will not build sample 'simpleVulkan'")
    endif()
else()
    message(STATUS "Vulkan not found - will not build sample 'simpleVulkan'")
endif()
