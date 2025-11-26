cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/Modules")

project(cudaNvSciNvMedia LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 87 110)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../../Common)

find_package(NVSCI)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Find the NVSCI/NVMEDIA libraries
    # use CMAKE_LIBRARY_PATH so that users can also specify the NVSCI lib path in cmake command
    set(CMAKE_LIBRARY_PATH "/usr/lib" ${CMAKE_LIBRARY_PATH})
    foreach(LIBRARY_PATH ${CMAKE_LIBRARY_PATH})
        file(GLOB_RECURSE NVMEDIA_LIB
            ${LIBRARY_PATH}/libnvmedia.so
            ${LIBRARY_PATH}/*/libnvmedia.so
        )
        if(NVMEDIA_LIB)
            break()
        endif()
    endforeach()

    # Find the NVSCI/NVMEDIA header files
    # use CMAKE_INCLUDE_PATH so that users can also specify the NVSCI/NVMEDIA include path in cmake command
    set(CMAKE_INCLUDE_PATH
        "/usr/include"
        "/usr/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/include"
        ${CMAKE_LIBRARY_PATH}
    )
    find_path(NVMEDIA_INCLUDE_DIR nvmedia_core.h PATHS ${CMAKE_INCLUDE_PATH})

    if(NVSCI_FOUND)
        if(NVMEDIA_LIB AND NVMEDIA_INCLUDE_DIR)
            message(STATUS "FOUND NVMEDIA libs: ${NVMEDIA_LIB}")
            message(STATUS "Using NVMEDIA headers path: ${NVMEDIA_INCLUDE_DIR}")
            # Source file
            # Add target for cudaNvSciNvMedia
            add_executable(cudaNvSciNvMedia imageKernels.cu cudaNvSciNvMedia.cpp main.cpp)

            target_compile_options(cudaNvSciNvMedia PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

            target_compile_features(cudaNvSciNvMedia PRIVATE cxx_std_17 cuda_std_17)

            set_target_properties(cudaNvSciNvMedia PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

            target_include_directories(cudaNvSciNvMedia PUBLIC
                ${CUDAToolkit_INCLUDE_DIRS}
                ${NVSCI_INCLUDE_DIRS}
                ${NVMEDIA_INCLUDE_DIR}
            )

            target_link_libraries(cudaNvSciNvMedia
                CUDA::cuda_driver
                ${NVSCI_LIBRARIES}
                ${NVMEDIA_LIB}
            )
            # Copy teapot.rgba to the output directory
            add_custom_command(TARGET cudaNvSciNvMedia POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/teapot.rgba ${CMAKE_CURRENT_BINARY_DIR}/teapot.rgba
            )

            # Specify additional clean files
            set_target_properties(cudaNvSciNvMedia PROPERTIES
                ADDITIONAL_CLEAN_FILES "teapot_out.rgba"
            )
        else()
            message(STATUS "NvMedia not found - will not build sample 'cudaNvSciNvMedia'")
        endif()
    else()
        message(STATUS "NvSCI not found - will not build sample 'cudaNvSciNvMedia'")
    endif()
else()
    message(STATUS "Will not build sample cudaNvSciNvMedia - requires Linux OS")
endif()
