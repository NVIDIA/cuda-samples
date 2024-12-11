# Include directories and libraries
include_directories(../../../Common)

find_package(OpenGL)
find_package(GLUT)

# Source file
set(SRC_FILES
    simpleCUDA2GL.cu
    main.cpp
)


if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        # Add target for simpleCUDA2GL
        add_executable(simpleCUDA2GL ${SRC_FILES})
        set_target_properties(simpleCUDA2GL PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(simpleCUDA2GL PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(simpleCUDA2GL
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )
    else()
        message(STATUS "GLUT not found - will not build sample 'simpleCUDA2GL'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'simpleCUDA2GL'")
endif()
