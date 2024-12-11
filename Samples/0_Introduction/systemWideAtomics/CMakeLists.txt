# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    systemWideAtomics.cu
)

# Add target for systemWideAtomics
add_executable(systemWideAtomics ${SRC_FILES})
set_target_properties(systemWideAtomics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
