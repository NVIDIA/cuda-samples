# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    conjugateGradientMultiDeviceCG.cu
)

# Add target for conjugateGradientMultiDeviceCG
add_executable(conjugateGradientMultiDeviceCG ${SRC_FILES})
set_target_properties(conjugateGradientMultiDeviceCG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(conjugateGradientMultiDeviceCG PRIVATE
    CUDA::cublas
    CUDA::cusparse
)
