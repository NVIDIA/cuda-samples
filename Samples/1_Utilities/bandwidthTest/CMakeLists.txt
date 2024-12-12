# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    bandwidthTest.cu
)

# Add target for bandwidthTest
add_executable(bandwidthTest ${SRC_FILES})
set_target_properties(bandwidthTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
