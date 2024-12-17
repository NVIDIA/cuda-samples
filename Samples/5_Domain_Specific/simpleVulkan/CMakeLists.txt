cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(simpleVulkan LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

find_package(Vulkan)
find_package(OpenGL)

# Source file
if(${Vulkan_FOUND})
    if(${OPENGL_FOUND})
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
            glfw
        )

        add_custom_command(TARGET simpleVulkan POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/sinewave.frag
            ${CMAKE_CURRENT_SOURCE_DIR}/sinewave.vert
            ${CMAKE_CURRENT_SOURCE_DIR}/vert.spv
            ${CMAKE_CURRENT_SOURCE_DIR}/frag.spv
            ${CMAKE_CURRENT_BINARY_DIR}
        )
    else()
        message(STATUS "GLFW not found - will not build sample 'simpleVulkan'")
    endif()
else()
    message(STATUS "Vulkan not found - will not build sample 'simpleVulkan'")
endif()
