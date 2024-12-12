# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    segmentationTree.cu
)

# Add target for segmentationTreeThrust
add_executable(segmentationTreeThrust ${SRC_FILES})
set_target_properties(segmentationTreeThrust PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
