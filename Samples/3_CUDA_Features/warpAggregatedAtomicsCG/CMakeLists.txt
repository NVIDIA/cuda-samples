# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    warpAggregatedAtomicsCG.cu
)

# Add target for warpAggregatedAtomicsCG
add_executable(warpAggregatedAtomicsCG ${SRC_FILES})
set_target_properties(warpAggregatedAtomicsCG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
