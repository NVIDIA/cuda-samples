# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    cudaTensorCoreGemm.cu
)

# Add target for cudaTensorCoreGemm
add_executable(cudaTensorCoreGemm ${SRC_FILES})
set_target_properties(cudaTensorCoreGemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
