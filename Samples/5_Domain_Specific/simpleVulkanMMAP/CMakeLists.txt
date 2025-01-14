cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleVulkanMMAP LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

find_package(Vulkan)
find_package(OpenGL)

# Include the check_include_file macro
include(CheckIncludeFile)

# Check for the GLFW/glfw3.h header
check_include_file("GLFW/glfw3.h" HAVE_GLFW3_H)

# Source file
if(${Vulkan_FOUND})
    if(${OPENGL_FOUND})
        if(${HAVE_GLFW3_H})
            # Add target for simpleVulkanMMAP
            add_executable(simpleVulkanMMAP ../../../Common/helper_multiprocess.cpp MonteCarloPi.cu VulkanBaseApp.cpp main.cpp)

            target_compile_options(simpleVulkanMMAP PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

            target_compile_features(simpleVulkanMMAP PRIVATE cxx_std_17 cuda_std_17)

            set_target_properties(simpleVulkanMMAP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

            target_include_directories(simpleVulkanMMAP PUBLIC
                ${Vulkan_INCLUDE_DIRS}
                ${CUDAToolkit_INCLUDE_DIRS}
            )

            target_link_libraries(simpleVulkanMMAP
                ${Vulkan_LIBRARIES}
                OpenGL::GL
                CUDA::cuda_driver
                glfw
            )

            add_custom_command(TARGET simpleVulkanMMAP POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/montecarlo.frag
                ${CMAKE_CURRENT_SOURCE_DIR}/montecarlo.vert
                ${CMAKE_CURRENT_SOURCE_DIR}/vert.spv
                ${CMAKE_CURRENT_SOURCE_DIR}/frag.spv
                ${CMAKE_CURRENT_BINARY_DIR}
            )
        else()
            message(STATUS "glfw3 not found - will not build sample 'simpleVulkanMMAP'")
        endif()
    else()
        message(STATUS "GLFW not found - will not build sample 'simpleVulkanMMAP'")
    endif()
else()
    message(STATUS "Vulkan not found - will not build sample 'simpleVulkanMMAP'")
endif()
