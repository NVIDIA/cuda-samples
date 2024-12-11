# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleAttributes.cu
)

# Add target for simpleAttributes
add_executable(simpleAttributes ${SRC_FILES})
set_target_properties(simpleAttributes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
