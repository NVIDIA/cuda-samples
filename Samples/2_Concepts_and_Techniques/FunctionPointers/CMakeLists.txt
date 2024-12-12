# Include directories and libraries
include_directories(../../../Common)


find_package(OpenGL)
find_package(GLUT)

# Source file
set(SRC_FILES
    FunctionPointers.cpp
    FunctionPointers_kernels.cu
)

if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        # Add target for FunctionPointers
        add_executable(FunctionPointers ${SRC_FILES})
        set_target_properties(FunctionPointers PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(FunctionPointers PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(FunctionPointers
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )

        # Copy clock_kernel.cu to the output directory
        add_custom_command(TARGET FunctionPointers POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}
        )

    else()
        message(STATUS "GLUT not found - will not build sample 'FunctionPointers'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'FunctionPointers'")
endif()
