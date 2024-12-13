# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    cuSolverRf.cpp
    mmio.c
    mmio_wrapper.cpp
)

# Add target for cuSolverRf
add_executable(cuSolverRf ${SRC_FILES})
set_target_properties(cuSolverRf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cuSolverRf PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cuSolverRf PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

# Copy data files to output directory
add_custom_command(TARGET cuSolverRf POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/*.mtx
    ${CMAKE_CURRENT_BINARY_DIR}
)
