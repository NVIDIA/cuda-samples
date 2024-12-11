# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    asyncAPI.cu
)

# Add target for asyncAPI
add_executable(asyncAPI ${SRC_FILES})
set_target_properties(asyncAPI PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
