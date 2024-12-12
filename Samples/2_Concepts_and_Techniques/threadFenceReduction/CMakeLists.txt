# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    threadFenceReduction.cu
)

# Add target for threadFenceReduction
add_executable(threadFenceReduction ${SRC_FILES})
set_target_properties(threadFenceReduction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
