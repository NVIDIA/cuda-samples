cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(watershedSegmentationNPP LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "native")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for watershedSegmentationNPP
add_executable(watershedSegmentationNPP watershedSegmentationNPP.cpp)

target_compile_options(watershedSegmentationNPP PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(watershedSegmentationNPP PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(watershedSegmentationNPP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(watershedSegmentationNPP PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(watershedSegmentationNPP PRIVATE
        CUDA::nppc
        CUDA::nppisu
        CUDA::nppif
        CUDA::cudart
)
