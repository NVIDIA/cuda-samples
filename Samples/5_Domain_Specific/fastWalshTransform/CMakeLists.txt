# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    fastWalshTransform.cu
    fastWalshTransform_gold.cpp
)

# Add target for fastWalshTransform
add_executable(fastWalshTransform ${SRC_FILES})
set_target_properties(fastWalshTransform PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
