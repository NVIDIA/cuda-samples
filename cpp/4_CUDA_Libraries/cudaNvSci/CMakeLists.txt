cmake_minimum_required(VERSION 3.20)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(cudaNvSci LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

find_package(NVSCI)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "QNX")
    if(NVSCI_FOUND)
        message(STATUS "FOUND NVSCI libs: ${NVSCIBUF_LIB} ${NVSCISYNC_LIB}")
        message(STATUS "Using NVSCI headers path: ${NVSCIBUF_INCLUDE_DIR} ${NVSCIBUF_INCLUDE_DIR}")
        add_executable(cudaNvSci imageKernels.cu cudaNvSci.cpp main.cpp)
        target_compile_options(cudaNvSci PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        target_compile_features(cudaNvSci PRIVATE cxx_std_17 cuda_std_17)
        target_include_directories(cudaNvSci PUBLIC
            ${CUDAToolkit_INCLUDE_DIRS}
            ${NVSCI_INCLUDE_DIRS}
        )
        target_link_libraries(cudaNvSci
            CUDA::cuda_driver
            ${NVSCI_LIBRARIES}
        )
        # Copy teapot1024.ppm to the output directory
        add_custom_command(TARGET cudaNvSci POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/teapot1024.ppm ${CMAKE_CURRENT_BINARY_DIR}/teapot1024.ppm
        )
        # Specify additional clean files
        set_target_properties(cudaNvSci PROPERTIES
            ADDITIONAL_CLEAN_FILES "teapot1024_out.ppm"
        )
        include(InstallSamples)
        setup_samples_install()
    else()
        message(STATUS "NvSCI not found - will not build sample 'cudaNvSci'")
    endif()
else()
    message(STATUS "Will not build sample cudaNvSci - requires Linux or QNX")
endif()
