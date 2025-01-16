cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(cuSolverSp_LinearSolver LANGUAGES C CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for cuSolverSp_LinearSolver
add_executable(cuSolverSp_LinearSolver cuSolverSp_LinearSolver.cpp mmio.c mmio_wrapper.cpp)

target_compile_options(cuSolverSp_LinearSolver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cuSolverSp_LinearSolver PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(cuSolverSp_LinearSolver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cuSolverSp_LinearSolver PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cuSolverSp_LinearSolver PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

# Copy data files to output directory
add_custom_command(TARGET cuSolverSp_LinearSolver POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/lap2D_5pt_n100.mtx
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Copy data files to output directory
add_custom_command(TARGET cuSolverSp_LinearSolver POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/lap3D_7pt_n20.mtx
    ${CMAKE_CURRENT_BINARY_DIR}
)
