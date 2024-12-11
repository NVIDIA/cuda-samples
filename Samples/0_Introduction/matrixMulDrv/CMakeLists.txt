# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    matrixMulDrv.cu
)

# Add target for matrixMulDrv
add_executable(matrixMulDrv ${SRC_FILES})
set_target_properties(matrixMulDrv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
