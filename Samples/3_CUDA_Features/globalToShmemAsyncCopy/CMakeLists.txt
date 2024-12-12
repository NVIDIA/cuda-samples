# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    globalToShmemAsyncCopy.cu
)

# Add target for globalToShmemAsyncCopy
add_executable(globalToShmemAsyncCopy ${SRC_FILES})
set_target_properties(globalToShmemAsyncCopy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
