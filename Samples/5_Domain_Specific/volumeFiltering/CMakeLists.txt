# Include directories and libraries
include_directories(../../../Common)

find_package(OpenGL)
find_package(GLUT)

# Source file
set(SRC_FILES
    volume.cpp
    volumeFilter_kernel.cu
    volumeFiltering.cpp
    volumeRender_kernel.cu
)

if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        # Add target for volumeFiltering
        add_executable(volumeFiltering ${SRC_FILES})
        set_target_properties(volumeFiltering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(volumeFiltering PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(volumeFiltering
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )

        add_custom_command(TARGET volumeFiltering POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
        )

    else()
        message(STATUS "GLUT not found - will not build sample 'volumeFiltering'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'volumeFiltering'")
endif()
