# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    binaryPartitionCG.cu
)

# Add target for binaryPartitionCG
add_executable(binaryPartitionCG ${SRC_FILES})
set_target_properties(binaryPartitionCG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
