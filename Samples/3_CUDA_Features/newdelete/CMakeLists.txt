# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    newdelete.cu
)

# Add target for newdelete
add_executable(newdelete ${SRC_FILES})
set_target_properties(newdelete PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
