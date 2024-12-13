# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    cuSolverDn_LinearSolver.cpp
    mmio.c
    mmio_wrapper.cpp
)

# Add target for cuSolverDn_LinearSolver
add_executable(cuSolverDn_LinearSolver ${SRC_FILES})
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
