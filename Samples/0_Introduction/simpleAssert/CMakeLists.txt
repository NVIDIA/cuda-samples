# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleAssert.cu
)

# Add target for simpleAssert
add_executable(simpleAssert ${SRC_FILES})
set_target_properties(simpleAssert PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
