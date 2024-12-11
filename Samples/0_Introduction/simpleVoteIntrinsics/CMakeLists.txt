# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleVoteIntrinsics.cu
)

# Add target for simpleVoteIntrinsics
add_executable(simpleVoteIntrinsics ${SRC_FILES})
set_target_properties(simpleVoteIntrinsics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
