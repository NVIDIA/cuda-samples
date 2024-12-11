# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleP2P.cu
)

# Add target for simpleP2P
add_executable(simpleP2P ${SRC_FILES})
set_target_properties(simpleP2P PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
