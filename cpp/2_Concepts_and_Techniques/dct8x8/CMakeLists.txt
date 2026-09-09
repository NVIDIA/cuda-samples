cmake_minimum_required(VERSION 3.20)


include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

if(SAMPLE_SKIP_BUILD)
    return()
endif()

project(dct8x8 LANGUAGES C CXX CUDA)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

find_package(CUDAToolkit REQUIRED)

include_directories(../../../Common)

add_executable(dct8x8 dct8x8.cu BmpUtil.cpp DCT8x8_Gold.cpp)

target_compile_options(dct8x8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(dct8x8 PRIVATE cxx_std_17 cuda_std_17)

target_include_directories(dct8x8 PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)

file(GLOB REF_DATA "teapot512*")
add_custom_command(TARGET dct8x8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${REF_DATA}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Copy data files to output directory
add_custom_command(TARGET dct8x8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/teapot512.ppm
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Copy data files to output directory
add_custom_command(TARGET dct8x8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/teapot512.bmp
    ${CMAKE_CURRENT_BINARY_DIR}/
)

include(InstallSamples)
setup_samples_install()
