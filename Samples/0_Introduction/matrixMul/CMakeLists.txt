# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    matrixMul.cu
)

# Add target for asyncAPI
add_executable(matrixMul ${SRC_FILES})
set_target_properties(matrixMul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
