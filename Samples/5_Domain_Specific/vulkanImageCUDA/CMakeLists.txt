# Include directories and libraries
include_directories(../../../Common)

find_package(Vulkan)
find_package(OpenGL)

# Source file
set(SRC_FILES
    vulkanImageCUDA.cu
)

if(${Vulkan_FOUND})
    if(${OPENGL_FOUND})
        # Add target for vulkanImageCUDA
        add_executable(vulkanImageCUDA ${SRC_FILES})
        set_target_properties(vulkanImageCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(vulkanImageCUDA PUBLIC
            ${Vulkan_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
        )

        target_link_libraries(vulkanImageCUDA
            ${Vulkan_LIBRARIES}
            OpenGL::GL
            glfw
        )

        add_custom_command(TARGET vulkanImageCUDA POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag
            ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert
            ${CMAKE_CURRENT_SOURCE_DIR}/vert.spv
            ${CMAKE_CURRENT_SOURCE_DIR}/frag.spv
            ${CMAKE_CURRENT_SOURCE_DIR}/teapot1024.ppm
            ${CMAKE_CURRENT_BINARY_DIR}
        )
    else()
        message(STATUS "GLFW not found - will not build sample 'vulkanImageCUDA'")
    endif()
else()
    message(STATUS "Vulkan not found - will not build sample 'vulkanImageCUDA'")
endif()
