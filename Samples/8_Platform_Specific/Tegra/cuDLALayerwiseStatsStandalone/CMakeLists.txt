cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(cuDLALayerwiseStatsStandalone LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 53 61 70 72 75 80 86 87 90)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../../Common)

find_library(CUDLA_LIB cudla PATHS ${CUDAToolkit_LIBRARY_DIR})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CUDLA_LIB)
        # Find the NVSCI libraries
        # use CMAKE_LIBRARY_PATH so that users can also specify the NVSCI lib path in cmake command
        set(CMAKE_LIBRARY_PATH "/usr/lib" ${CMAKE_LIBRARY_PATH})
        file(GLOB_RECURSE NVSCIBUF_LIB
            ${CMAKE_LIBRARY_PATH}/*/libnvscibuf.so
        )
        file(GLOB_RECURSE NVSCISYNC_LIB
            ${CMAKE_LIBRARY_PATH}/*/libnvscisync.so
        )

        # Find the NVSCI header files
        # use CMAKE_INCLUDE_PATH so that users can also specify the NVSCI include path in cmake command
        set(CMAKE_INCLUDE_PATH "/usr/include" ${CMAKE_LIBRARY_PATH})
        find_path(NVSCIBUF_INCLUDE_DIR nvscibuf.h PATHS ${CMAKE_INCLUDE_PATH})
        find_path(NVSCISYNC_INCLUDE_DIR nvscisync.h PATHS ${CMAKE_INCLUDE_PATH})

        if(NVSCIBUF_LIB AND NVSCISYNC_LIB AND NVSCIBUF_INCLUDE_DIR AND NVSCISYNC_INCLUDE_DIR)
            # Source file
            # Add target for cuDLALayerwiseStatsStandalone
            add_executable(cuDLALayerwiseStatsStandalone main.cpp)

            target_compile_options(cuDLALayerwiseStatsStandalone PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

            target_compile_features(cuDLALayerwiseStatsStandalone PRIVATE cxx_std_17 cuda_std_17)

            set_target_properties(cuDLALayerwiseStatsStandalone PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

            target_include_directories(cuDLALayerwiseStatsStandalone PUBLIC
                ${CUDAToolkit_INCLUDE_DIRS}
                ${NVSCIBUF_INCLUDE_DIR}
                ${NVSCISYNC_INCLUDE_DIR}
            )

            target_link_libraries(cuDLALayerwiseStatsStandalone
                ${CUDLA_LIB}
                ${NVSCIBUF_LIB}
                ${NVSCISYNC_LIB}
            )
        else()
            message(STATUS "NvSCI not found - will not build sample 'cuDLALayerwiseStatsStandalone'")
        endif()
    else()
        message(STATUS "CUDLA not found - will not build sample 'cuDLALayerwiseStatsStandalone'")
    endif()
else()
    message(STATUS "Will not build sample cuDLALayerwiseStatsStandalone - requires Linux OS")
endif()
