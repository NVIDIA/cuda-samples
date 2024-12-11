# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleMultiGPU.cu
)

# Add target for simpleMultiGPU
add_executable(simpleMultiGPU ${SRC_FILES})
set_target_properties(simpleMultiGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
