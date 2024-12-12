# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    reduction.cpp
    reduction_kernel.cu
)

# Add target for reduction
add_executable(reduction ${SRC_FILES})
set_target_properties(reduction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(reduction PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)
