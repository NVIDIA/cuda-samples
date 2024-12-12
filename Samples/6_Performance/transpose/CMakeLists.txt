# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    transpose.cu
)

# Add target for transpose
add_executable(transpose ${SRC_FILES})
set_target_properties(transpose PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
