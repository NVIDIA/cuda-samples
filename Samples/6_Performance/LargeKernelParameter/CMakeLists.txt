# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    LargeKernelParameter.cu
)

# Add target for LargeKernelParameter
add_executable(LargeKernelParameter ${SRC_FILES})
set_target_properties(LargeKernelParameter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
