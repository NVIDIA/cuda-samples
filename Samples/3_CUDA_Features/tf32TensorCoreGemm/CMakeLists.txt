# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    tf32TensorCoreGemm.cu
)

# Add target for tf32TensorCoreGemm
add_executable(tf32TensorCoreGemm ${SRC_FILES})
set_target_properties(tf32TensorCoreGemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
