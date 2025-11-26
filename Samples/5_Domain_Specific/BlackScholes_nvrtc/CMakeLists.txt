cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(BlackScholes_nvrtc LANGUAGES C CXX CUDA)

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
# Add target for BlackScholes_nvrtc
add_executable(BlackScholes_nvrtc BlackScholes.cpp BlackScholes_gold.cpp)

target_compile_options(BlackScholes_nvrtc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(BlackScholes_nvrtc PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(BlackScholes_nvrtc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(BlackScholes_nvrtc PRIVATE
    CUDA::nvrtc
    CUDA::cuda_driver
)

# Copy kernel to the output directory
add_custom_command(TARGET BlackScholes_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/BlackScholes_kernel.cuh ${CMAKE_CURRENT_BINARY_DIR}
)
