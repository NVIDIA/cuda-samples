# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleZeroCopy.cu
)

# Add target for simpleZeroCopy
add_executable(simpleZeroCopy ${SRC_FILES})
set_target_properties(simpleZeroCopy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
