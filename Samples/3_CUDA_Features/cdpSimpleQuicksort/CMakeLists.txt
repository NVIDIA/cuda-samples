# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    cdpSimpleQuicksort.cu
)

# Add target for cdpSimpleQuicksort
add_executable(cdpSimpleQuicksort ${SRC_FILES})
set_target_properties(cdpSimpleQuicksort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
