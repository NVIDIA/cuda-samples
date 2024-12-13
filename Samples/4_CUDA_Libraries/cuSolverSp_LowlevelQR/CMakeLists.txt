# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    cuSolverSp_LowlevelQR.cpp
    mmio.c
    mmio_wrapper.cpp
)

# Add target for cuSolverSp_LowlevelQR
add_executable(cuSolverSp_LowlevelQR ${SRC_FILES})
set_target_properties(cuSolverSp_LowlevelQR PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cuSolverSp_LowlevelQR PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cuSolverSp_LowlevelQR PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

# Copy data files to output directory
add_custom_command(TARGET cuSolverSp_LowlevelQR POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/*.mtx
    ${CMAKE_CURRENT_BINARY_DIR}
)
