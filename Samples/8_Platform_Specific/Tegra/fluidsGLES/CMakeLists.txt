cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(fluidsGLES LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package (OpenGL COMPONENTS EGL GLES2)
    if(${OpenGL_EGL_FOUND})
        # Source file
        # Add target for fluidsGLES
        add_executable(fluidsGLES 
            fluidsGLES.cpp
            fluidsGLES_kernels.cu
        )

        target_compile_options(fluidsGLES PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(fluidsGLES PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(fluidsGLES PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(fluidsGLES PUBLIC
            ${OPENGL_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
            ../../../Common
        )

        target_link_libraries(fluidsGLES
            ${OPENGL_LIBRARIES}
            CUDA::cuda_driver
        )
    else()
        message(STATUS "EGL not found - will not build sample 'fluidsGLES'")
    endif()
else()
    message(STATUS "Will not build sample fluidsGLES - requires Linux OS")
endif()
