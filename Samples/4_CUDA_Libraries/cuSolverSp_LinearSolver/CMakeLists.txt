# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    cuSolverSp_LinearSolver.cpp
    mmio.c
    mmio_wrapper.cpp
)

# Add target for cuSolverSp_LinearSolver
add_executable(cuSolverSp_LinearSolver ${SRC_FILES})
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
    ${CMAKE_CURRENT_SOURCE_DIR}/*.mtx
    ${CMAKE_CURRENT_BINARY_DIR}
)
