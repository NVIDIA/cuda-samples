# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    streamOrderedAllocation.cu
)

# Add target for streamOrderedAllocation
add_executable(streamOrderedAllocation ${SRC_FILES})
set_target_properties(streamOrderedAllocation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
