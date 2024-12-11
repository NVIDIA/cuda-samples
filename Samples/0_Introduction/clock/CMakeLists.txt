# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    clock.cu
)

# Add target for asyncAPI
add_executable(clock ${SRC_FILES})
set_target_properties(clock PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
