# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    cdpAdvancedQuicksort.cu
    cdpBitonicSort.cu
)

# Add target for cdpAdvancedQuicksort
add_executable(cdpAdvancedQuicksort ${SRC_FILES})
set_target_properties(cdpAdvancedQuicksort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
