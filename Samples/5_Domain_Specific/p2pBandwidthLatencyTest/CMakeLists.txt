# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    p2pBandwidthLatencyTest.cu
)

# Add target for p2pBandwidthLatencyTest
add_executable(p2pBandwidthLatencyTest ${SRC_FILES})
set_target_properties(p2pBandwidthLatencyTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
