# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    dmmaTensorCoreGemm.cu
)

# Add target for dmmaTensorCoreGemm
add_executable(dmmaTensorCoreGemm ${SRC_FILES})
set_target_properties(dmmaTensorCoreGemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
