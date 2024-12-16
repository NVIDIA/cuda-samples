# Include directories and libraries
include_directories(../../../Common)

find_package(Vulkan)
find_package(OpenGL)

# Source file
set(SRC_FILES
    ../../../Common/helper_multiprocess.cpp
    MonteCarloPi.cu
    VulkanBaseApp.cpp
    main.cpp
)

if(${Vulkan_FOUND})
    if(${OPENGL_FOUND})
        # Add target for simpleVulkanMMAP
        add_executable(simpleVulkanMMAP ${SRC_FILES})
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
        message(STATUS "GLFW not found - will not build sample 'simpleVulkanMMAP'")
    endif()
else()
    message(STATUS "Vulkan not found - will not build sample 'simpleVulkanMMAP'")
endif()
