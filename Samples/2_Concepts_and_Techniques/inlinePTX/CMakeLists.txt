# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    inlinePTX.cu
)

# Add target for inlinePTX
add_executable(inlinePTX ${SRC_FILES})
set_target_properties(inlinePTX PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
