# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    fp16ScalarProduct.cu
)

# Add target for asyncAPI
add_executable(fp16ScalarProduct ${SRC_FILES})
set_target_properties(fp16ScalarProduct PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
