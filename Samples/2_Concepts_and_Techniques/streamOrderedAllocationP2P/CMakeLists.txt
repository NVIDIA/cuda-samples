# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    streamOrderedAllocationP2P.cu
)

# Add target for streamOrderedAllocationP2P
add_executable(streamOrderedAllocationP2P ${SRC_FILES})
set_target_properties(streamOrderedAllocationP2P PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
