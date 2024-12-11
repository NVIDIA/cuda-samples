# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    vectorAdd.cu
)

# Add target for vectorAdd
add_executable(vectorAdd ${SRC_FILES})
set_target_properties(vectorAdd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
