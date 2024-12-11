# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    simpleAWBarrier.cu
)

# Add target for simpleAWBarrier
add_executable(simpleAWBarrier ${SRC_FILES})
set_target_properties(simpleAWBarrier PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
