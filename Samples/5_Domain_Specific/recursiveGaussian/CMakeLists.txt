# Include directories and libraries
include_directories(../../../Common)

find_package(OpenGL)
find_package(GLUT)

# Source file
set(SRC_FILES
    recursiveGaussian_cuda.cu
    recursiveGaussian.cpp
)

if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        # Add target for recursiveGaussian
        add_executable(recursiveGaussian ${SRC_FILES})
        set_target_properties(recursiveGaussian PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(recursiveGaussian PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(recursiveGaussian
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )

        add_custom_command(TARGET recursiveGaussian POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
        )

    else()
        message(STATUS "GLUT not found - will not build sample 'recursiveGaussian'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'recursiveGaussian'")
endif()
