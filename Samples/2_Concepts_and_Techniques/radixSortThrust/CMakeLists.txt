# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    radixSortThrust.cu
)

# Add target for radixSortThrust
add_executable(radixSortThrust ${SRC_FILES})
set_target_properties(radixSortThrust PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
