# Include directories and libraries
include_directories(../../../Common)

find_package(OpenGL)
find_package(GLUT)

# Source file
set(SRC_FILES
    GLSLProgram.cpp
    ParticleSystem.cpp
    ParticleSystem_cuda.cu
    SmokeRenderer.cpp
    SmokeShaders.cpp
    framebufferObject.cpp
    particleDemo.cpp
    renderbuffer.cpp
)

if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        # Add target for smokeParticles
        add_executable(smokeParticles ${SRC_FILES})
        set_target_properties(smokeParticles PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(smokeParticles PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(smokeParticles
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )

        add_custom_command(TARGET smokeParticles POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
        )

    else()
        message(STATUS "GLUT not found - will not build sample 'smokeParticles'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'smokeParticles'")
endif()
