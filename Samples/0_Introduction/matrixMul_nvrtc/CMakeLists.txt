# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    matrixMul_nvrtc.cu
)

# Add target for matrixMul_nvrtc
add_executable(matrixMul_nvrtc ${SRC_FILES})
set_target_properties(matrixMul_nvrtc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
