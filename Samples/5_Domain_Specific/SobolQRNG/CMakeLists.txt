# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    sobol_gold.cpp
    sobol_gpu.cu
    sobol_primitives.cpp
    sobol.cpp
)

# Add target for SobolQRNG
add_executable(SobolQRNG ${SRC_FILES})
set_target_properties(SobolQRNG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(SobolQRNG PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)
