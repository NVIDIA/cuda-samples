cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(binomialOptions_nvrtc LANGUAGES C CXX CUDA)

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

# Source file
# Add target for binomialOptions_nvrtc
add_executable(binomialOptions_nvrtc binomialOptions.cpp binomialOptions_gold.cpp binomialOptions_gpu.cpp)

target_compile_options(binomialOptions_nvrtc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(binomialOptions_nvrtc PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(binomialOptions_nvrtc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(binomialOptions_nvrtc PRIVATE
    CUDA::nvrtc
    CUDA::cuda_driver
)

# Copy kernel to the output directory
add_custom_command(TARGET binomialOptions_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/binomialOptions_kernel.cu ${CMAKE_CURRENT_BINARY_DIR}
)

# Copy header to the output directory
add_custom_command(TARGET binomialOptions_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/common_gpu_header.h ${CMAKE_CURRENT_BINARY_DIR}
)

# Copy header to the output directory
add_custom_command(TARGET binomialOptions_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/binomialOptions_common.h ${CMAKE_CURRENT_BINARY_DIR}
)

# Copy header to the output directory
add_custom_command(TARGET binomialOptions_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/realtype.h ${CMAKE_CURRENT_BINARY_DIR}
)
