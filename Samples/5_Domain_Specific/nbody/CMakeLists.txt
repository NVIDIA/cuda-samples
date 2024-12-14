# Include directories and libraries
include_directories(../../../Common)

find_package(OpenGL)
find_package(GLUT)

# Source file
set(SRC_FILES
    bodysystemcuda.cu
    render_particles.cpp
    nbody.cpp
)

if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        # Add target for nbody
        add_executable(nbody ${SRC_FILES})
        set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(nbody PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(nbody
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )

    else()
        message(STATUS "GLUT not found - will not build sample 'nbody'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'nbody'")
endif()
