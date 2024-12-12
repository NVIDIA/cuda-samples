# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    graphMemoryFootprint.cu
)

# Add target for graphMemoryFootprint
add_executable(graphMemoryFootprint ${SRC_FILES})
set_target_properties(graphMemoryFootprint PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
