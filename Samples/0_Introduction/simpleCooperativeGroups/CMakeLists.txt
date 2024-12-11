# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleCooperativeGroups.cu
)

# Add target for simpleCooperativeGroups
add_executable(simpleCooperativeGroups ${SRC_FILES})
set_target_properties(simpleCooperativeGroups PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
