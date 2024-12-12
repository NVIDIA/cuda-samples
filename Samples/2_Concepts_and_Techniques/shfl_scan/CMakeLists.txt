# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    shfl_scan.cu
)

# Add target for shfl_scan
add_executable(shfl_scan ${SRC_FILES})
set_target_properties(shfl_scan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
