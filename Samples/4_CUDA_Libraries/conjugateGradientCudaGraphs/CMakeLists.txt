# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    conjugateGradientCudaGraphs.cu
)

# Add target for conjugateGradientCudaGraphs
add_executable(conjugateGradientCudaGraphs ${SRC_FILES})
set_target_properties(conjugateGradientCudaGraphs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(conjugateGradientCudaGraphs PRIVATE
    CUDA::cublas
    CUDA::cusparse
)
