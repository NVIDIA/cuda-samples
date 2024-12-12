# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    graphConditionalNodes.cu
)

# Add target for graphConditionalNodes
add_executable(graphConditionalNodes ${SRC_FILES})
set_target_properties(graphConditionalNodes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
