# Include directories and libraries
include_directories(../../../Common)

find_package(OpenGL)
find_package(GLUT)

# Source file
set(SRC_FILES
    volumeRender_kernel.cu
    volumeRender.cpp
)

if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        # Add target for volumeRender
        add_executable(volumeRender ${SRC_FILES})
        set_target_properties(volumeRender PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(volumeRender PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(volumeRender
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )

        add_custom_command(TARGET volumeRender POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
        )

    else()
        message(STATUS "GLUT not found - will not build sample 'volumeRender'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'volumeRender'")
endif()
