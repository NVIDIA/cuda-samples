# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    cudaGraphPerfScaling.cu
)

# Add target for cudaGraphsPerfScaling
add_executable(cudaGraphsPerfScaling ${SRC_FILES})
set_target_properties(cudaGraphsPerfScaling PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
