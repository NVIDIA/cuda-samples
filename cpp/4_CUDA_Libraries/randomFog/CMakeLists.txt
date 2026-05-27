cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(randomFog LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Check for Debian13 or later
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    file(READ "/etc/os-release" OS_RELEASE_CONTENTS)
    if(OS_RELEASE_CONTENTS MATCHES "ID=[^ ]*debian[^ ]*" AND
       OS_RELEASE_CONTENTS MATCHES "VERSION_ID=\"(1[3-9]|[2-9][0-9])\"")
        message(STATUS "Debian 13 or later detected")
        set(DEBIAN TRUE)
    endif()
endif()

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

# Source file
if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        # Add target for randomFog
        add_executable(randomFog randomFog.cpp rng.cpp)

        target_compile_options(randomFog PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(randomFog PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(randomFog PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(randomFog PUBLIC
            ${OPENGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GLUT_INCLUDE_DIRS}
        )

        target_link_libraries(randomFog
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
            CUDA::curand
            CUDA::cudart
        )
        # Need to add X11 and other libraries for Debian13 or later explicitly
        if(DEBIAN)
            target_link_libraries(randomFog
                X11
                Xi
                Xxf86vm
                Xext
            )
        endif()

        # Copy data files to the output directory
        add_custom_command(TARGET randomFog POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
        )

        if(WIN32)
            target_link_libraries(randomFog
                ${PC_GLUT_LIBRARY_DIRS}/freeglut.lib
                ${PC_GLUT_LIBRARY_DIRS}/${GLEW_LIB_NAME}.lib
            )

            add_custom_command(TARGET randomFog
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/../../../bin/win64/$<CONFIGURATION>/freeglut.dll
                ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>
            )

            add_custom_command(TARGET randomFog
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/../../../bin/win64/$<CONFIGURATION>/${GLEW_LIB_NAME}.dll
                ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>
            )
        endif()

    else()
        message(STATUS "GLUT not found - will not build sample 'randomFog'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'randomFog'")
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
