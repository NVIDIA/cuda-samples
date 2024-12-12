# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    StreamPriorities.cu
)

# Add target for StreamPriorities
add_executable(StreamPriorities ${SRC_FILES})
set_target_properties(StreamPriorities PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
