# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    reductionMultiBlockCG.cu
)

# Add target for reductionMultiBlockCG
add_executable(reductionMultiBlockCG ${SRC_FILES})
set_target_properties(reductionMultiBlockCG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
