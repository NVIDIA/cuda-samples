cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(cuSolverDn_LinearSolver LANGUAGES C CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for cuSolverDn_LinearSolver
add_executable(cuSolverDn_LinearSolver cuSolverDn_LinearSolver.cpp mmio.c mmio_wrapper.cpp)

target_compile_options(cuSolverDn_LinearSolver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cuSolverDn_LinearSolver PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(cuSolverDn_LinearSolver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cuSolverDn_LinearSolver PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cuSolverDn_LinearSolver PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

# Copy data files to output directory
add_custom_command(TARGET cuSolverDn_LinearSolver POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/*.mtx
    ${CMAKE_CURRENT_BINARY_DIR}
)
