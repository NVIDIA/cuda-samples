# Include directories and libraries
include_directories(../../../Common)

find_package(OpenGL)
find_package(GLUT)

# Source file
set(SRC_FILES
    simpleTexture3D_kernel.cu
    simpleTexture3D.cpp
)

if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        # Add target for simpleTexture3D
        add_executable(simpleTexture3D ${SRC_FILES})
        set_target_properties(simpleTexture3D PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(simpleTexture3D PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(simpleTexture3D
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )

        # Copy clock_kernel.cu to the output directory
        add_custom_command(TARGET simpleTexture3D POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}
        )

    else()
        message(STATUS "GLUT not found - will not build sample 'simpleTexture3D'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'simpleTexture3D'")
endif()
