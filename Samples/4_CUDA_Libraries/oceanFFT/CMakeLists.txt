# Include directories and libraries
include_directories(../../../Common)

find_package(OpenGL)
find_package(GLUT)

# Source file
set(SRC_FILES
    oceanFFT.cpp
    oceanFFT_kernel.cu
)

if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        # Add target for oceanFFT
        add_executable(oceanFFT ${SRC_FILES})
        set_target_properties(oceanFFT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(oceanFFT PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(oceanFFT
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
            CUDA::cufft
        )

        # Copy clock_kernel.cu to the output directory
        add_custom_command(TARGET oceanFFT POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
        )

    else()
        message(STATUS "GLUT not found - will not build sample 'oceanFFT'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'oceanFFT'")
endif()
