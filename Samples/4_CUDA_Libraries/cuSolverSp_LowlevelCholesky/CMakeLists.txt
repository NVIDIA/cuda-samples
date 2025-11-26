cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(cuSolverSp_LowlevelCholesky LANGUAGES C CXX)

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
# Add target for cuSolverSp_LowlevelCholesky
add_executable(cuSolverSp_LowlevelCholesky cuSolverSp_LowlevelCholesky.cpp mmio.c mmio_wrapper.cpp)

target_compile_options(cuSolverSp_LowlevelCholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cuSolverSp_LowlevelCholesky PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(cuSolverSp_LowlevelCholesky PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cuSolverSp_LowlevelCholesky PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cuSolverSp_LowlevelCholesky PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

# Copy data files to output directory
add_custom_command(TARGET cuSolverSp_LowlevelCholesky POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/lap2D_5pt_n100.mtx
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Copy data files to output directory
add_custom_command(TARGET cuSolverSp_LowlevelCholesky POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/lap3D_7pt_n20.mtx
    ${CMAKE_CURRENT_BINARY_DIR}
)
