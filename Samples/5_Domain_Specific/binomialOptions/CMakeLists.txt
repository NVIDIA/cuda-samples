# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    binomialOptions.cpp
    binomialOptions_gold.cpp
    binomialOptions_kernel.cu
)

# Add target for binomialOptions
add_executable(binomialOptions ${SRC_FILES})
set_target_properties(binomialOptions PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(binomialOptions PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)
