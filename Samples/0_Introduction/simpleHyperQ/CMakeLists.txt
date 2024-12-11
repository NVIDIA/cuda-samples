# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleHyperQ.cu
)

# Add target for simpleHyperQ
add_executable(simpleHyperQ ${SRC_FILES})
set_target_properties(simpleHyperQ PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
