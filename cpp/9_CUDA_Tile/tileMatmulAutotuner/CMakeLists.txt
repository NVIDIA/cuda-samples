cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(tileMatmulAutotuner LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

get_filename_component(CUDA_TOOLKIT_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
find_program(TILEIRAS_EXECUTABLE tileiras
    HINTS "${CUDA_TOOLKIT_BIN_DIR}"
    REQUIRED
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --enable-tile")

if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common ../Benchmark_Common)

# Source files
add_executable(tileMatmulAutotuner
    matmul_autotuner.cpp
    autotuner_search_space.conf
)

target_compile_features(tileMatmulAutotuner PRIVATE cxx_std_20 cuda_std_20)

target_include_directories(tileMatmulAutotuner PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_INCLUDE_DIR)
file(TO_CMAKE_PATH "${CUDA_INCLUDE_DIR}" CUDA_INCLUDE_DIR_FOR_DEFINE)
file(TO_CMAKE_PATH "${TILEIRAS_EXECUTABLE}" TILEIRAS_EXECUTABLE_FOR_DEFINE)
file(TO_CMAKE_PATH "${CMAKE_CUDA_COMPILER}" NVCC_EXECUTABLE_FOR_DEFINE)

target_compile_definitions(tileMatmulAutotuner PRIVATE
    CUDA_INCLUDE_PATH="${CUDA_INCLUDE_DIR_FOR_DEFINE}"
    NVCC_PATH="${NVCC_EXECUTABLE_FOR_DEFINE}"
    TILEIRAS_PATH="${TILEIRAS_EXECUTABLE_FOR_DEFINE}"
)

target_link_libraries(tileMatmulAutotuner PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::nvrtc
)

add_custom_command(TARGET tileMatmulAutotuner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cu
    ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/autotuner_search_space.conf
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
