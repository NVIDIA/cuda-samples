# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleStreams.cu
)

# Add target for simpleStreams
add_executable(simpleStreams ${SRC_FILES})
set_target_properties(simpleStreams PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
