# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    cudaNvSciNvMedia.cu
)

# Add target for cudaNvSciNvMedia
add_executable(cudaNvSciNvMedia ${SRC_FILES})
set_target_properties(cudaNvSciNvMedia PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
