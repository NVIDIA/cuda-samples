# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    conjugateGradientMultiBlockCG.cu
)

# Add target for conjugateGradientMultiBlockCG
add_executable(conjugateGradientMultiBlockCG ${SRC_FILES})
set_target_properties(conjugateGradientMultiBlockCG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(conjugateGradientMultiBlockCG PRIVATE
    CUDA::cublas
    CUDA::cusparse
)
