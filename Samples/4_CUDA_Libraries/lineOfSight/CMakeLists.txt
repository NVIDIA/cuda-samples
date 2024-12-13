# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    lineOfSight.cu
)

# Add target for lineOfSight
add_executable(lineOfSight ${SRC_FILES})
set_target_properties(lineOfSight PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
