# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simplePrintf.cu
)

# Add target for simplePrintf
add_executable(simplePrintf ${SRC_FILES})
set_target_properties(simplePrintf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
