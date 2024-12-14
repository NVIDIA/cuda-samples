# Include directories and libraries
include_directories(../../../Common)

find_package(OpenGL)
find_package(GLUT)

# Source file
set(SRC_FILES
    bicubicTexture.cpp
    bicubicTexture_cuda.cu
)

if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        # Add target for bicubicTexture
        add_executable(bicubicTexture ${SRC_FILES})
        set_target_properties(bicubicTexture PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(bicubicTexture PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(bicubicTexture
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )

        # Copy clock_kernel.cu to the output directory
        add_custom_command(TARGET bicubicTexture POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
        )

    else()
        message(STATUS "GLUT not found - will not build sample 'bicubicTexture'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'bicubicTexture'")
endif()
