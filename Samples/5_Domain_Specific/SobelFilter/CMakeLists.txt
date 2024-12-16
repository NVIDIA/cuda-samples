# Include directories and libraries
include_directories(../../../Common)

find_package(OpenGL)
find_package(GLUT)

# Source file
set(SRC_FILES
    SobelFilter.cpp
    SobelFilter_kernels.cu
)

if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        # Add target for SobelFilter
        add_executable(SobelFilter ${SRC_FILES})
        set_target_properties(SobelFilter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(SobelFilter PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(SobelFilter
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )

        add_custom_command(TARGET SobelFilter POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
        )

    else()
        message(STATUS "GLUT not found - will not build sample 'SobelFilter'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'SobelFilter'")
endif()
