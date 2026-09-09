cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(volumeRender LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

if(WIN32)
    set(PC_GLUT_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../Common")
    set(PC_GLUT_LIBRARY_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../Common/lib/x64")
    # The GLEW library built on Windows on Arm system is named as glew32.lib/glew32.dll by default.
    if(EXISTS "${PC_GLUT_LIBRARY_DIRS}/glew32.lib")
        set(GLEW_LIB_NAME "glew32")
    else()
        set(GLEW_LIB_NAME "glew64")
    endif()
endif()

find_package(OpenGL)
find_package(GLUT)

if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        add_executable(volumeRender volumeRender_kernel.cu volumeRender.cpp)
        target_compile_options(volumeRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        target_compile_features(volumeRender PRIVATE cxx_std_17 cuda_std_17)
        target_include_directories(volumeRender PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )
        target_link_libraries(volumeRender
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
        )
        add_custom_command(TARGET volumeRender POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
        )
        if(WIN32)
            target_link_libraries(volumeRender
                ${PC_GLUT_LIBRARY_DIRS}/freeglut.lib
                ${PC_GLUT_LIBRARY_DIRS}/${GLEW_LIB_NAME}.lib
            )
            add_custom_command(TARGET volumeRender
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/../../../bin/win64/$<CONFIGURATION>/freeglut.dll
                ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>
            )
            add_custom_command(TARGET volumeRender
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/../../../bin/win64/$<CONFIGURATION>/${GLEW_LIB_NAME}.dll
                ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>
            )
        endif()
        include(InstallSamples)
        setup_samples_install()
    else()
        message(STATUS "GLUT not found - will not build sample 'volumeRender'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'volumeRender'")
endif()
