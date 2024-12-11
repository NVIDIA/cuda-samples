# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleMultiCopy.cu
)

# Add target for simpleMultiCopy
add_executable(simpleMultiCopy ${SRC_FILES})
set_target_properties(simpleMultiCopy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
