# Include directories and libraries
include_directories(../../../Common)

# Source file
set(SRC_FILES
    streamOrderedAllocationIPC.cu
    ../../../Common/helper_multiprocess.cpp
)

# Add target for streamOrderedAllocationIPC
add_executable(streamOrderedAllocationIPC ${SRC_FILES})
set_target_properties(streamOrderedAllocationIPC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(streamOrderedAllocationIPC PUBLIC
    CUDA::cuda_driver
)
