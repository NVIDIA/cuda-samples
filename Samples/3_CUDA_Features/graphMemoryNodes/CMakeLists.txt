# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    graphMemoryNodes.cu
)

# Add target for graphMemoryNodes
add_executable(graphMemoryNodes ${SRC_FILES})
set_target_properties(graphMemoryNodes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
