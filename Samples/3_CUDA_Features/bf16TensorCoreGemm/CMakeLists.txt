# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    bf16TensorCoreGemm.cu
)

# Add target for bf16TensorCoreGemm
add_executable(bf16TensorCoreGemm ${SRC_FILES})
set_target_properties(bf16TensorCoreGemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
