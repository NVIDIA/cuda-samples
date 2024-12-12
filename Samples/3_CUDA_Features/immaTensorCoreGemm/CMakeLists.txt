# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    immaTensorCoreGemm.cu
)

# Add target for immaTensorCoreGemm
add_executable(immaTensorCoreGemm ${SRC_FILES})
set_target_properties(immaTensorCoreGemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
