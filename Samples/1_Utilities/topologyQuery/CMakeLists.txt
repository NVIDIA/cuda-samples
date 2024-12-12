# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    topologyQuery.cu
)

# Add target for topologyQuery
add_executable(topologyQuery ${SRC_FILES})
set_target_properties(topologyQuery PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
