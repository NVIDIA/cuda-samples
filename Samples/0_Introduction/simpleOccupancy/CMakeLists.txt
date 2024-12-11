# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleOccupancy.cu
)

# Add target for simpleOccupancy
add_executable(simpleOccupancy ${SRC_FILES})
set_target_properties(simpleOccupancy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
