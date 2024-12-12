# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleCudaGraphs.cu
)

# Add target for simpleCudaGraphs
add_executable(simpleCudaGraphs ${SRC_FILES})
set_target_properties(simpleCudaGraphs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
