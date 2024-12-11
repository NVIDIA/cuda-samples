# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleTemplates.cu
)

# Add target for simpleTemplates
add_executable(simpleTemplates ${SRC_FILES})
set_target_properties(simpleTemplates PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
