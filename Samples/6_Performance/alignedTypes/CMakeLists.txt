# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    alignedTypes.cu
)

# Add target for alignedTypes
add_executable(alignedTypes ${SRC_FILES})
set_target_properties(alignedTypes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
