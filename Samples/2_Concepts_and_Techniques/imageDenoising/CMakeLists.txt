# Include directories and libraries
include_directories(../../../Common)

find_package(OpenGL)
find_package(GLUT)

# Source file
set(SRC_FILES
    bmploader.cpp
    imageDenoising.cu
    imageDenoisingGL.cpp
)

if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        # Add target for imageDenoising
        add_executable(imageDenoising ${SRC_FILES})
        set_target_properties(imageDenoising PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(imageDenoising PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(imageDenoising
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )

        # Copy clock_kernel.cu to the output directory
        add_custom_command(TARGET imageDenoising POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
        )

    else()
        message(STATUS "GLUT not found - will not build sample 'imageDenoising'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'imageDenoising'")
endif()
