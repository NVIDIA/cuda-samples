cmake_minimum_required(VERSION 3.20)

set(SAMPLE_DISALLOW_ARCHS 75 80 86 89 90 100 103 107 120)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(cudaNvSciBufMultiplanar LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../../Common)

find_package(NVSCI)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(NVSCI_FOUND)
        add_executable(cudaNvSciBufMultiplanar imageKernels.cu cudaNvSciBufMultiplanar.cpp main.cpp)
        target_compile_options(cudaNvSciBufMultiplanar PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        target_compile_features(cudaNvSciBufMultiplanar PRIVATE cxx_std_17 cuda_std_17)
        target_include_directories(cudaNvSciBufMultiplanar PUBLIC
            ${CUDAToolkit_INCLUDE_DIRS}
            ${NVSCI_INCLUDE_DIRS}
        )
        target_link_libraries(cudaNvSciBufMultiplanar
            CUDA::cuda_driver
            ${NVSCI_LIBRARIES}
        )
        # Copy yuv_planar_img1.yuv to the output directory
        add_custom_command(TARGET cudaNvSciBufMultiplanar POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/yuv_planar_img1.yuv ${CMAKE_CURRENT_BINARY_DIR}/yuv_planar_img1.yuv
        )
        # Specify additional clean files
        set_target_properties(cudaNvSciBufMultiplanar PROPERTIES
            ADDITIONAL_CLEAN_FILES "image_out.yuv"
        )
        include(InstallSamples)
        setup_samples_install()
    else()
        message(STATUS "NvSCI not found - will not build sample 'cudaNvSciBufMultiplanar'")
    endif()
else()
    message(STATUS "Will not build sample cudaNvSciBufMultiplanar - requires Linux OS")
endif()
