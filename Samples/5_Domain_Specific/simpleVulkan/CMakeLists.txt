# Include directories and libraries
include_directories(../../../Common)

find_package(Vulkan)
find_package(OpenGL)

# Source file
set(SRC_FILES
    main.cpp
    SineWaveSimulation.cu
    VulkanBaseApp.cpp
)

if(${Vulkan_FOUND})
    if(${OPENGL_FOUND})
        # Add target for simpleVulkan
        add_executable(simpleVulkan ${SRC_FILES})
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
